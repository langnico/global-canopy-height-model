import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, BatchSampler, Sampler, RandomSampler, SequentialSampler, SubsetRandomSampler
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
import wandb

from gchm.utils.transforms import denormalize
from gchm.utils.loss import filter_nans_from_tensors, get_classification_metrics_lookup
from gchm.utils.sampler import SliceBatchSampler, SubsetSequentialSampler


DEVICE = torch.device("cuda:0")
print('DEVICE: ', DEVICE, torch.cuda.get_device_name(0))
INF = torch.tensor(float('-inf')).to(DEVICE)
NAN = torch.tensor(float('nan')).to(DEVICE)


def nan2neginf(x):
    return torch.where(torch.isnan(x), INF, x)


def neginf2nan(x):
    return torch.where(torch.isfinite(x), x,  NAN)


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class Trainer:
    def __init__(self, model, args,
                 ds_train, ds_val,
                 metrics_lookup,
                 train_input_mean, train_input_std,
                 train_target_mean, train_target_std,
                 sample_weighted_loss=None):

        print('DEVICE: ', DEVICE)

        self.eval_step = 500  # compute train metrics every eval_step iterations

        self.model = model
        self.args = args
        self.writer = SummaryWriter(log_dir=args.log_dir)

        self.ds_train = ds_train
        self.ds_val = ds_val

        self.metrics_lookup = metrics_lookup

        #self.train_input_mean = train_input_mean
        #self.train_input_std = train_input_std
        self.train_target_mean = torch.tensor(train_target_mean).to(DEVICE)
        self.train_target_std = torch.tensor(train_target_std).to(DEVICE)

        self.sample_weighted_loss = sample_weighted_loss
        
        self.classification_loss_names = list(get_classification_metrics_lookup().keys())  # to check if targets need to be converted to long()
        
        if self.args.max_pool_predictions:
            print('Predictions are max pooled before supervision')
            self.max_pool_pred = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, return_indices=True)

        if self.args.max_pool_labels:
            print('Labels are max pooled before supervision')
            self.max_pool_label = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1, return_indices=True)

        self.optimizer = self._setup_optimizer()
        self.checkpoint_path = Path(self.args.out_dir) / 'checkpoint.pt'

        self.scheduler = self._setup_scheduler()

    def _setup_optimizer(self):
        #if self.args.freeze_features:
        #    params_to_optimize = list(self.model.predictions.parameters()) + list( self.model.variances.parameters())
        #else:
        #    params_to_optimize = self.model.parameters()
        
        params_to_optimize = self.model.parameters()

        if self.args.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(params_to_optimize, lr=self.args.base_learning_rate,
                                         weight_decay=self.args.l2_lambda)

        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params_to_optimize, lr=self.args.base_learning_rate,
                                        weight_decay=self.args.l2_lambda)
        else:
            raise ValueError("Optimizer '{}' is not implemented in _setup_optimizer().".format(self.args.optimizer))
        return optimizer

    def _setup_scheduler(self):
        if self.args.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.lr_milestones,
                                                             gamma=0.1)
            print('Scheduler MultiStepLR: Learning rate will be dropped at epochs: {}'.format(self.args.lr_milestones))

        elif self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                            max_lr=self.args.base_learning_rate,
                                                            epochs=self.args.nb_epoch,
                                                            steps_per_epoch=self.args.iterations_per_epoch)
            print('Scheduler OneCycleLR: args.base_learning_rate is used as max_lr: {}'.format(self.args.base_learning_rate))
        else:
            raise ValueError("Scheduler '{}' is not implemented in _setup_scheduler().".format(self.args.scheduler))
        return scheduler

    def train(self):
        # Initialize train and validation loader
        print('self.args.custom_sampler: ', self.args.custom_sampler)
        if self.args.custom_sampler is None:
            dl_train = DataLoader(self.ds_train, batch_size=self.args.batch_size, shuffle=True,
                                  num_workers=self.args.num_workers, pin_memory=True)

            # The val set is shuffled to use only a random subset after each epoch for validation (speed up)
            dl_val = DataLoader(self.ds_val, batch_size=self.args.batch_size, shuffle=True,
                                num_workers=self.args.num_workers, pin_memory=True)

        elif self.args.custom_sampler == 'SliceBatchSampler':
            # Note this implementation expects that the data was shuffled in advance.
            dl_train = DataLoader(dataset=self.ds_train, batch_size=None,
                                  sampler=SliceBatchSampler(sampler=SubsetRandomSampler(range(0, len(self.ds_train), self.args.batch_size)),   #RandomSampler(self.ds_train),   #range(0, len(self.ds_train), self.args.batch_size),
                                                            batch_size=self.args.batch_size,
                                                            slice_step=self.args.slice_step,
                                                            num_samples=len(self.ds_train),
                                                            drop_last=False),
                                  num_workers=self.args.num_workers, pin_memory=True)

            dl_val = DataLoader(dataset=self.ds_val, batch_size=None,
                                sampler=SliceBatchSampler(sampler=SubsetRandomSampler(range(0, len(self.ds_val), self.args.batch_size)),   #  RandomSampler(self.ds_val),   #range(0, len(self.ds_val), self.args.batch_size),
                                                          batch_size=self.args.batch_size,
                                                          slice_step=self.args.slice_step,
                                                          num_samples=len(self.ds_val),
                                                          drop_last=False),
                                num_workers=self.args.num_workers, pin_memory=True)
        
        elif self.args.custom_sampler == 'BatchSampler':
            dl_train = DataLoader(dataset=self.ds_train, batch_size=None,
                                  sampler=BatchSampler(sampler=RandomSampler(self.ds_train),
                                                            batch_size=self.args.batch_size,
                                                            drop_last=False),
                                  num_workers=self.args.num_workers, pin_memory=True)

            # The val set is shuffled to use only a random subset after each epoch for validation (speed up)
            dl_val = DataLoader(dataset=self.ds_val, batch_size=None,
                                sampler=BatchSampler(sampler=RandomSampler(self.ds_val),
                                                          batch_size=self.args.batch_size,
                                                          drop_last=False),
                                num_workers=self.args.num_workers, pin_memory=True)
        else:
            raise(ValueError, "This custom sampler type is not implemented: ", self.args.custom_sampler)

        # Init best losses for weights saving.
        loss_val_best = np.inf
        best_epoch = None

        # load pre-trained model weights
        if self.args.model_weights_path is not None:
            print('ATTENTION: loading pre-trained model weights from:')
            print(self.args.model_weights_path)
            checkpoint = torch.load(self.args.model_weights_path)
            model_weights = checkpoint['model_state_dict']
            self.model.load_state_dict(model_weights)

        # load checkpoint if exists
        if self.checkpoint_path.exists():
            print('Found existing checkpoint...')
            checkpoint = torch.load(self.checkpoint_path)

            # load and update existing model_state_dict (model weights)
            # the update step is needed when the model contains more layers then the loaded pretrained model weights
            model_dict = self.model.state_dict()
            model_dict.update(checkpoint['model_state_dict'])
            self.model.load_state_dict(model_dict)

            epoch_start = checkpoint['epoch']
            self.scheduler.load_state_dict(checkpoint['scheduler'])

            if self.args.load_optimizer_state_dict:
                print("Loading checkpoint['optimizer_state_dict']")
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Checkpoint from epoch {} is loaded. val_loss={:.3f}'.format(epoch_start, checkpoint['val_loss']))
        else:
            epoch_start = 0

        if self.args.reinit_last_layer:
            print('Re-initializing last layer...')
            self.model.predictions.reset_parameters()
            self.model.variances.reset_parameters()

        # Start training
        for epoch in range(epoch_start, self.args.nb_epoch):

            epoch += 1
            print('Epoch: {} / {} '.format(epoch, self.args.nb_epoch))

            # optimize parameters
            training_metrics = self.optimize_epoch(dl_train, epoch)

            # validated performance
            print('validating...')
            if self.args.iterations_per_epoch:
                # set number of val iterations to 20 % of number of train iterations per epoch
                num_val_iterations = int(self.args.iterations_per_epoch * 0.2)
            else:
                num_val_iterations = len(dl_val)
            val_dict, val_metrics = self.validate(dl_val, num_iterations=num_val_iterations)

            if self.scheduler and self.args.scheduler not in ['OneCycleLR']:
                self.scheduler.step()

            metric_keys = list(training_metrics.keys())
            # -------- LOG TRAINING METRICS --------
            # tensorboard
            metric_string = 'TRAIN: '
            for metric in metric_keys:
                # tensorboard logs
                self.writer.add_scalar('{}/train'.format(metric), training_metrics[metric], epoch)
                metric_string += ' {}: {:.3f},'.format(metric, training_metrics[metric])
            print(metric_string)

            # wandb
            training_metrics_wandb = {}
            for key in list(training_metrics.keys()):
                new_key = "{}_{}".format("train", key)
                training_metrics_wandb[new_key] = training_metrics[key]
            wandb.log(training_metrics_wandb, step=epoch)

            # -------- LOG VALIDATION METRICS --------
            # tensorboard
            metric_string = 'VAL:   '
            for metric in metric_keys:
                # tensorboard logs
                self.writer.add_scalar('{}/val'.format(metric), val_metrics[metric], epoch)
                metric_string += ' {}: {:.3f},'.format(metric, val_metrics[metric])
            print(metric_string)

            # wandb
            val_metrics_wandb = {}
            for key in list(val_metrics.keys()):
                new_key = "{}_{}".format("val", key)
                val_metrics_wandb[new_key] = val_metrics[key]
            wandb.log(val_metrics_wandb, step=epoch)

            # logging the estimated variance
            if 'variances' in val_dict:
                if self.args.normalize_targets:
                    # denormalize the variance to log in original units
                    val_dict['variances'] = val_dict['variances'] * self.train_target_std.cpu() ** 2

                self.writer.add_scalar('RMV/val', torch.sqrt(torch.mean(val_dict['variances'])), epoch)
                self.writer.add_scalar('var_mean/val', torch.mean(val_dict['variances']), epoch)
                self.writer.add_scalar('std_mean/val', torch.mean(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('std_min/val', torch.min(torch.sqrt(val_dict['variances'])), epoch)
                self.writer.add_scalar('std_max/val', torch.max(torch.sqrt(val_dict['variances'])), epoch)
                if self.args.debug:
                    self.writer.add_scalar('var_count_infinite_elements/val',
                                           self.count_infinite_elements(val_dict['variances']), epoch)

                    print('VAL DEBUG: Number of infinite elements in variances: ',
                          self.count_infinite_elements(val_dict['variances']))
                    print("val_dict['variances']", val_dict['variances'])

            if val_metrics[self.args.loss_key] < loss_val_best:
                loss_val_best = val_metrics[self.args.loss_key]
                best_epoch = epoch
                # save and overwrite the best model weights:
                path = Path(self.args.out_dir) / 'best_weights.pt'
                torch.save(self.model.state_dict(), path)
                print('Saved weights at {}'.format(path))

            # stop training if loss is nan
            if not self.args.debug:
                if np.isnan(training_metrics[self.args.loss_key]) or np.isnan(val_metrics[self.args.loss_key]):
                    raise ValueError("Training loss is nan. Stop training.")

            # save checkpoint
            print('saving checkpoint...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_metrics[self.args.loss_key],
                'scheduler': self.scheduler.state_dict()},
                self.checkpoint_path)

        # TODO: Currently we save only the best weights --> maybe want to add every nth epoch.
        print('Best val loss: {} at epoch: {}'.format(loss_val_best, best_epoch))
        # save model weights after last epoch:
        path = Path(self.args.out_dir) / 'weights_last_epoch.pt'
        torch.save(self.model.state_dict(), path)
        print('Saved weights at {}'.format(path))

    def optimize_epoch(self, dl_train, epoch):

        # setup profiler context manager in debug mode.
        # else use an empty context manager with no effect.
        if self.args.do_profile:
            profiler_context_manager = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                                              schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
                                                              on_trace_ready=torch.profiler.tensorboard_trace_handler(self.args.log_dir))
        else:
            profiler_context_manager = nullcontext()  # this will set profiler=None

        with profiler_context_manager as profiler:

            if not self.args.iterations_per_epoch:
                self.args.iterations_per_epoch = len(dl_train)

            # init running error
            training_metrics = {}
            for metric in list(self.metrics_lookup.keys()) + ['loss']:
                training_metrics[metric] = 0

            total_count_infinite_var = 0
            count_eval_steps = 0
            self.model.train()
            with tqdm(total=self.args.iterations_per_epoch, ncols=100, desc='train') as pbar:
                for step, data_dict in enumerate(dl_train):

                    if step == self.args.iterations_per_epoch:
                        break

                    inputs, labels = data_dict[self.args.input_key], data_dict[self.args.label_mean_key]
                    inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                    
                    # do not compute grads for input (should reduce the graph and speed up training when freeze_features=True)
                    inputs.requires_grad_(False)

                    if self.sample_weighted_loss:
                        weights = data_dict[self.args.weight_key]
                        weights = weights.to(DEVICE, non_blocking=True)
                        weights = weights + torch.tensor(float(self.args.eps)).to(DEVICE)

                    if self.args.max_pool_labels:
                        labels = nan2neginf(labels)
                        labels, max_indices = self.max_pool_label(labels)
                        labels = neginf2nan(labels)
                        if self.sample_weighted_loss:
                            weights = retrieve_elements_from_indices(weights, max_indices)

                    if self.args.return_variance:
                        # Run forward pass
                        predictions, variances = self.model.forward(inputs)
                        if self.args.max_pool_predictions:
                            predictions, max_indices = self.max_pool_pred(predictions)
                            variances = retrieve_elements_from_indices(variances, max_indices)

                        if self.sample_weighted_loss:
                            [predictions, variances, labels, weights] = filter_nans_from_tensors(tensors=[predictions, variances, labels, weights], mask_src=labels)
                            loss = self.sample_weighted_loss(predictions, labels, weights, variances)
                        else:
                            [predictions, variances, labels] = filter_nans_from_tensors(tensors=[predictions, variances, labels], mask_src=labels)
                            loss = self.metrics_lookup[self.args.loss_key](predictions, variances, labels)

                        # check variances
                        if self.args.debug:
                            count_infinite = self.count_infinite_elements(variances)
                            total_count_infinite_var += count_infinite
                    else:
                        # Run forward pass
                        predictions = self.model.forward(inputs)
                        if self.args.max_pool_predictions:
                            predictions, _ = self.max_pool_pred(predictions)

                        if self.sample_weighted_loss:
                            [predictions, labels, weights] = filter_nans_from_tensors(tensors=[predictions, labels, weights], mask_src=labels)
                        else:
                            [predictions, labels] = filter_nans_from_tensors(tensors=[predictions, labels], mask_src=labels)

                        # Prob. this was needed for classification (resp. semantic segmentation)
                        predictions = predictions.squeeze()
                        labels = labels.squeeze()

                        if self.args.loss_key in self.classification_loss_names:
                            labels = labels.long()

                        if self.sample_weighted_loss:
                            loss = self.sample_weighted_loss(predictions, labels, weights)
                        else:
                            loss = self.metrics_lookup[self.args.loss_key](predictions, labels)

                    # Run backward pass
                    self.optimizer.zero_grad()
                    loss.backward()

                    # optional gradient clippling (total norm of all parameters)
                    if self.args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    # optional gradient clippling (individual gradient values)
                    if self.args.max_grad_value:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.max_grad_value)

                    if self.args.debug:
                        # log gradients
                        iteration = (epoch-1) * self.args.iterations_per_epoch + step
                        parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
                        grad_norms = torch.stack([torch.norm(p.grad.detach(), 2).to(DEVICE) for p in parameters])
                        total_norm = torch.norm(grad_norms, 2.0).item()
                        grad_norm_mean = torch.mean(grad_norms).item()
                        grad_norm_max = torch.max(grad_norms).item()
                        grad_norm_min = torch.min(grad_norms).item()

                        self.writer.add_scalar('grad_norm/total', total_norm, iteration)
                        self.writer.add_scalar('grad_norm/mean', grad_norm_mean, iteration)
                        self.writer.add_scalar('grad_norm/max', grad_norm_max, iteration)
                        self.writer.add_scalar('grad_norm/min', grad_norm_min, iteration)
                        print('GRAD NORMS iteration: {}: total: {:.3f}, mean: {:.3f}, max: {:.3f}, min: {:.3f}'.format(iteration, total_norm, grad_norm_mean, grad_norm_max, grad_norm_min))

                        grad_values_abs = torch.cat([torch.abs(p.grad.data.flatten().detach().to(DEVICE)) for p in parameters])
                        grad_values_abs_mean = torch.mean(grad_values_abs).item()
                        grad_values_abs_max = torch.max(grad_values_abs).item()
                        grad_values_abs_min = torch.min(grad_values_abs).item()

                        self.writer.add_scalar('grad_values_abs/mean', grad_values_abs_mean, iteration)
                        self.writer.add_scalar('grad_values_abs/max', grad_values_abs_max, iteration)
                        self.writer.add_scalar('grad_values_abs/min', grad_values_abs_min, iteration)
                        print('GRAD VALUES iteration: {}: mean: {:.3f}, max: {:.3f}, min: {:.3f}'.format(iteration, grad_values_abs_mean, grad_values_abs_max, grad_values_abs_min))

                    self.optimizer.step()

                    # One-cycle learning rate policy changes the learning rate after every batch (not every epoch)
                    if self.scheduler and self.args.scheduler in ['OneCycleLR']:
                        self.scheduler.step()

                    # compute metrics on every batch and add to running sum
                    # compute average loss and metrics on every eval_step batch and add to running sum
                    if step % self.eval_step == 0:
                        count_eval_steps += 1
                        for metric in self.metrics_lookup:
                            if self.args.return_variance and metric in ['GNLL', 'LNLL',]:
                                training_metrics[metric] += self.metrics_lookup[metric](predictions, variances, labels).item()
                            else:
                                if self.args.normalize_targets:
                                    # denormalize labels and predictions to obtain metrics in original units
                                    predictions_ = denormalize(predictions, self.train_target_mean, self.train_target_std)
                                    labels_ = denormalize(labels, self.train_target_mean, self.train_target_std)
                                    training_metrics[metric] += self.metrics_lookup[metric](predictions_, labels_).item()
                                else:
                                    training_metrics[metric] += self.metrics_lookup[metric](predictions, labels).item()
                        training_metrics['loss'] += loss.item()
                    pbar.update(1)
                    if self.args.do_profile:
                        profiler.step()

        # average over number of batches
        for metric in self.metrics_lookup.keys():
            training_metrics[metric] /= count_eval_steps

        # debug
        if self.args.debug:
            if total_count_infinite_var > 0:
                print('TRAIN DEBUG: ATTENTION: count infinite elements in variances is: {}'.format(
                    total_count_infinite_var))
                print("predictions", predictions)
                print("variances", variances)

        return training_metrics

    def validate(self, dl_val, num_iterations=None, compute_loss=True):
        self.model.eval()

        #
        if not num_iterations:
            num_iterations = len(dl_val)

        # init validation results for current epoch
        val_dict = {'predictions': [], 'targets': []}

        if self.args.return_variance:
            val_dict['variances'] = []

        if compute_loss and self.sample_weighted_loss:
            val_dict['weights'] = []

        with torch.no_grad():
            with tqdm(total=num_iterations, ncols=100, desc='val') as pbar:
                for step, data_dict in enumerate(dl_val):

                    if step == num_iterations:
                        break

                    inputs, labels = data_dict[self.args.input_key], data_dict[self.args.label_mean_key]
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    if compute_loss and self.sample_weighted_loss:
                        weights = data_dict[self.args.weight_key]
                        weights = weights.to(DEVICE, non_blocking=True)

                    if self.args.max_pool_labels:
                        labels = nan2neginf(labels)
                        labels, max_indices = self.max_pool_label(labels)
                        labels = neginf2nan(labels)
                        if self.sample_weighted_loss:
                            weights = retrieve_elements_from_indices(weights, max_indices)

                    if self.args.return_variance:
                        predictions, variances = self.model.forward(inputs)
                        #if self.args.max_pool_predictions:
                        #    predictions, max_indices = self.max_pool_pred(predictions)
                        #    variances = retrieve_elements_from_indices(variances, max_indices)

                        if compute_loss and self.sample_weighted_loss:
                            [predictions, variances, labels, weights] = filter_nans_from_tensors(tensors=[predictions, variances, labels, weights], mask_src=labels)
                        else:
                            [predictions, variances, labels] = filter_nans_from_tensors(tensors=[predictions, variances, labels], mask_src=labels)
                        val_dict['variances'].extend(list(variances.cpu()))
                    else:
                        predictions = self.model.forward(inputs)
                        #if self.args.max_pool_predictions:
                        #    predictions, _ = self.max_pool_pred(predictions)

                        if self.sample_weighted_loss:
                            [predictions, labels, weights] = filter_nans_from_tensors(tensors=[predictions, labels, weights], mask_src=labels)
                        else:
                            [predictions, labels] = filter_nans_from_tensors(tensors=[predictions, labels], mask_src=labels)
                    
                    predictions = predictions.squeeze()
                    labels = labels.squeeze()

                    if self.args.loss_key in self.classification_loss_names:
                        labels = labels.long()

                    val_dict['predictions'].extend(list(predictions.cpu()))
                    val_dict['targets'].extend(list(labels.cpu()))
                    if compute_loss and self.sample_weighted_loss:
                        val_dict['weights'].extend(list(weights.cpu()))
                    pbar.update(1)

            for key in val_dict.keys():
                if val_dict[key]:
                    val_dict[key] = torch.stack(val_dict[key], dim=0)
                    print("val_dict['{}'].shape: ".format(key), val_dict[key].shape)
        
        print("unique: val_dict['targets']", torch.unique(val_dict['targets']))
        print("val_dict['targets'].dtype", val_dict['targets'].dtype)

        val_metrics = {}

        if compute_loss:
            if self.sample_weighted_loss:
                if self.args.return_variance:
                    loss = self.sample_weighted_loss(val_dict['predictions'], val_dict['targets'], val_dict['weights'], val_dict['variances'])
                else:
                    loss = self.sample_weighted_loss(val_dict['predictions'], val_dict['targets'], val_dict['weights'])
            else:
                if self.args.loss_key in ['GNLL', 'LNLL']:
                    loss = self.metrics_lookup[self.args.loss_key](val_dict['predictions'],
                                                                   val_dict['variances'],
                                                                   val_dict['targets'])
                else:
                    loss = self.metrics_lookup[self.args.loss_key](val_dict['predictions'], val_dict['targets'])
            val_metrics['loss'] = loss.item()

        for metric in self.metrics_lookup:
            if self.args.return_variance and metric in ['GNLL', 'LNLL']:
                val_metrics[metric] = self.metrics_lookup[metric](val_dict['predictions'],
                                                                  val_dict['variances'],
                                                                  val_dict['targets']).item()
            else:
                # denormalize labels and predictions
                if self.args.normalize_targets:
                    predictions_ = denormalize(val_dict['predictions'], self.train_target_mean.cpu(), self.train_target_std.cpu())
                    targets_ = denormalize(val_dict['targets'], self.train_target_mean.cpu(), self.train_target_std.cpu())
                    val_metrics[metric] = self.metrics_lookup[metric](predictions_, targets_).item()
                else:
                    val_metrics[metric] = self.metrics_lookup[metric](val_dict['predictions'],
                                                                      val_dict['targets']).item()
        return val_dict, val_metrics

    def test(self, model_weights=None, ds_test=None, batch_size=4096):
        if ds_test is None:
            ds_test = self.ds_val

        slice_start_indices = range(0, len(ds_test), batch_size)

        dl_test = DataLoader(dataset=ds_test, batch_size=None,
                             sampler=SliceBatchSampler(sampler=SubsetSequentialSampler(slice_start_indices),
                                                       batch_size=batch_size,
                                                       slice_step=self.args.slice_step,
                                                       num_samples=len(ds_test),
                                                       drop_last=False),
                             num_workers=self.args.num_workers, pin_memory=True)

        if model_weights is None:
            model_weights_path = self.checkpoint_path
            model_weights = torch.load(model_weights_path)

        # load best model weights
        self.model.load_state_dict(model_weights)

        test_dict, test_metrics = self.validate(dl_test, compute_loss=False)

        # denormalize predictions and targets
        if self.args.normalize_targets:
            test_dict['predictions'] = denormalize(test_dict['predictions'], self.train_target_mean.cpu(), self.train_target_std.cpu())
            test_dict['targets'] = denormalize(test_dict['targets'], self.train_target_mean.cpu(), self.train_target_std.cpu())
            if self.args.return_variance:
                # denormalize the variances by multiplying with the target variance
                test_dict['variances'] = test_dict['variances'] * self.train_target_std.cpu()**2

        if self.args.return_variance and self.args.debug:
            print('TEST: Number infinite elements in variances: ', self.count_infinite_elements(test_dict['variances']))

        # convert torch tensor to numpy
        for key in test_dict.keys():
            test_dict[key] = test_dict[key].data.cpu().numpy()

        metric_string = 'TEST:   '
        for metric in self.metrics_lookup:
            metric_string += ' {}: {:.3f},'.format(metric, test_metrics[metric])
        print(metric_string)
        return test_metrics, test_dict, metric_string

    def count_infinite_elements(self, x):
        return torch.sum(torch.logical_not(torch.isfinite(x))).item()


import datetime
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter  # to save the process of the model
from tqdm import tqdm
from collections import OrderedDict
import logging
from .networks import SWNN, STPNN, STNN_SPNN
from .utils import OLS, DIAGNOSIS
from .datasets import baseDataset, predictDataset, STNKDE_Dataset


class GNNWR:
    r"""
    GNNWR(Geographically neural network weighted regression) is a model to address spatial non-stationarity in various domains with complex geographical processes,
    which comes from the paper `Geographically neural network weighted regression for the accurate estimation of spatial non-stationarity <https://doi.org/10.1080/13658816.2019.1707834>`__.

    Parameters
    ----------
    train_dataset : baseDataset
        the dataset of training
    valid_dataset : baseDataset
        the dataset of validation
    test_dataset : baseDataset
        the dataset of testing
    dense_layers : list
        the dense layers of the model (default: ``None``)

        Default structure is a geometric sequence of power of 2, the minimum is 2, and the maximum is the power of 2 closest to the number of neurons in the input layer.

        i.e. ``[2,4,8,16,32,64,128,256]``
    start_lr : float
        the start learning rate of the model (default: ``0.1``)
    optimizer : str, optional
        the optimizer of the model (default: ``"Adagrad"``)
        choose from "SGD","Adam","RMSprop","Adagrad","Adadelta"
    drop_out : float
        the drop out rate of the model (default: ``0.2``)
    batch_norm : bool, optional
        whether use batch normalization (default: ``True``)
    activate_func : torch.nn
        the activate function of the model (default: ``nn.PReLU(init=0.4)``)
    model_name : str
        the name of the model (default: ``"GNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")``)
    model_save_path : str
        the path of the model (default: ``"../gnnwr_models"``)
    write_path : str
        the path of the log (default: ``"../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")``)
    use_gpu : bool
        whether use gpu or not (default: ``True``)
    use_ols : bool
        whether use ols or not (default: ``True``)
    log_path : str
        the path of the log (default: ``"../gnnwr_logs/"``)
    log_file_name : str
        the name of the log (default: ``"gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"``)
    log_level : int
        the level of the log (default: ``logging.INFO``)
    optimizer_params : dict, optional
        - **scheduler** \: *str*
            
            The type of learning rate scheduler to use. Valid options include ``Special``,
            ``Constant``, and ``MultiStepLR``.
        - **maxlr** \: *float*
            
            The maximum learning rate for the scheduler, for ``Special``.
        - **minlr** \: *float*
            
            The minimum learning rate for the scheduler, for ``Special``.
        - **upepoch** \: *int*
            
            The number of epochs until the maximum learning rate is reached, for ``Special``.
        - **decayepoch** \: *int*
            
            The epoch at which learning rate decay starts, for ``Special``.
        - **decayrate** \: *float*
            
            The rate at which the learning rate decays, for ``Special``.
        - **stop_change_epoch** \: *int*

            The epoch at which to stop adjusting the learning rate, for ``Special``.
        - **stop_lr** \: *float*

            The learning rate to stop at after the specified epoch, for ``Special``.
        - **scheduler_milestones** \: *list*

            The epochs at which to decay the learning rate for a ``MultiStepLR`` scheduler.
        - **scheduler_gamma** \: *float*

            The factor by which the learning rate is reduced for a ``MultiStepLR`` scheduler.
        - **weight_decay** \: *float*

            The weight decay factor for the optimizer.

    """

    def __init__(
            self,
            train_dataset,
            valid_dataset,
            test_dataset,
            dense_layers=None,
            start_lr: float = .1,
            optimizer="Adagrad",
            drop_out=0.2,
            batch_norm=True,
            activate_func=nn.PReLU(init=0.4),
            model_name="GNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
            model_save_path="../gnnwr_models",
            write_path="../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            use_gpu: bool = True,
            use_ols: bool = True,
            log_path="../gnnwr_logs/",
            log_file_name="gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
            log_level=logging.INFO,
            optimizer_params=None
    ):
        self._train_dataset = train_dataset  # train dataset
        self._valid_dataset = valid_dataset  # valid dataset
        self._test_dataset = test_dataset  # test dataset

        self._model_x_scale_info = train_dataset.x_scale_info
        self._model_y_scale_info = train_dataset.y_scale_info

        self._dense_layers = dense_layers  # structure of layers
        self._start_lr = start_lr  # initial learning rate
        self._insize = train_dataset.datasize  # size of input layer
        self._outsize = train_dataset.coefsize  # size of output layer
        self._writer = SummaryWriter(write_path)  # summary writer
        self._drop_out = drop_out  # drop_out ratio
        self._batch_norm = batch_norm  # batch normalization
        self._activate_func = activate_func  # activate function , default: PRelu(0.4)
        self._model = SWNN(self._dense_layers, self._insize, self._outsize,
                           self._drop_out, self._activate_func, self._batch_norm)  # model
        self._log_path = log_path  # log path
        self._log_file_name = log_file_name  # log file
        self._log_level = log_level  # log level
        self.__istrained = False  # whether the model is trained

        self._coefficient = OLS(
            train_dataset.scaledDataframe, train_dataset.x, train_dataset.y).params  # coefficients of OLS

        self._out = nn.Linear(
            self._outsize, 1, bias=False)  # layer to multiply OLS coefficients and model output
        if use_ols:
            self._out.weight = nn.Parameter(torch.tensor([self._coefficient]).to(
                torch.float32), requires_grad=False)  # define the weight
        else:
            self._coefficient = np.ones((1, self._outsize))
            self._out.weight = nn.Parameter(torch.tensor(np.ones((1, self._outsize))).to(
                torch.float32), requires_grad=False)  # define the weight

        self._criterion = nn.MSELoss()  # loss function
        self._trainLossList = []  # record the loss in training process
        self._validLossList = []  # record the loss in validation process
        self._epoch = 0  # current epoch
        self._bestr2 = float('-inf')  # best r2
        self._besttrainr2 = float('-inf')  # best train r2
        self._noUpdateEpoch = 0  # number of epochs without update
        # Model information
        self._modelName = model_name  # model name
        self._modelSavePath = model_save_path  # model save path
        self._train_diagnosis = None  # diagnosis of training
        self._test_diagnosis = None  # diagnosis of test
        self._valid_r2 = None  # r2 of validation
        self.result_data = None

        self._use_gpu = use_gpu
        if self._use_gpu:
            if torch.cuda.is_available():
                devices = [i for i in range(torch.cuda.device_count())]
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
            else:
                self._use_gpu = False

        self._optimizer = None
        self._scheduler = None
        self._optimizer_name = None
        self.init_optimizer(optimizer, optimizer_params)  # initialize the optimizer
        self._device = torch.device('cuda') if self._use_gpu else torch.device('cpu')

    def init_optimizer(self, optimizer, optimizer_params=None):
        r"""
        Initialize the optimizer.

        This method sets up the optimizer and the learning rate scheduler with the given parameters.

        Parameters
        ----------
        optimizer : str
            The name of the optimizer to use. This should be one of the supported optimizers.
        optimizer_params : dict, optional
            A dictionary containing parameters for the optimizer and scheduler. If not provided,
            default values will be used. The dictionary can contain the following keys:

            - **scheduler** \: *str*
            
            The type of learning rate scheduler to use. Valid options include ``Special``,
            ``Constant``, and ``MultiStepLR``.
            - **maxlr** \: *float*
                
                The maximum learning rate for the scheduler, for ``Special``.
            - **minlr** \: *float*
                
                The minimum learning rate for the scheduler, for ``Special``.
            - **upepoch** \: *int*
                
                The number of epochs until the maximum learning rate is reached, for ``Special``.
            - **decayepoch** \: *int*
                
                The epoch at which learning rate decay starts, for ``Special``.
            - **decayrate** \: *float*
                
                The rate at which the learning rate decays, for ``Special``.
            - **stop_change_epoch** \: *int*

                The epoch at which to stop adjusting the learning rate, for ``Special``.
            - **stop_lr** \: *float*

                The learning rate to stop at after the specified epoch, for ``Special``.
            - **scheduler_milestones** \: *list*

                The epochs at which to decay the learning rate for a ``MultiStepLR`` scheduler.
            - **scheduler_gamma** \: *float*

                The factor by which the learning rate is reduced for a ``MultiStepLR`` scheduler.
            - **weight_decay** \: *float*

                The weight decay factor for the optimizer.

        Raises
        ------
        ValueError
            If an unsupported optimizer or scheduler is specified.

        """
        # initialize the optimizer
        if optimizer_params is None:
            optimizer_params = {}
        weigth_decay = optimizer_params.get("weight_decay", 1e-3)
        
        if optimizer == "SGD":
            self._optimizer = optim.SGD(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "Adam":
            self._optimizer = optim.Adam(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "AdamW":
            self._optimizer = optim.AdamW(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "RMSprop":
            self._optimizer = optim.RMSprop(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "Adagrad":
            self._optimizer = optim.Adagrad(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        elif optimizer == "Adadelta":
            self._optimizer = optim.Adadelta(
                self._model.parameters(), lr=self._start_lr, weight_decay=weigth_decay)
        else:
            raise ValueError("Invalid Optimizer")
        self._optimizer_name = optimizer  # optimizer name

        # lr scheduler
        
        scheduler = optimizer_params.get("scheduler", "Constant")
        scheduler_milestones = optimizer_params.get(
            "scheduler_milestones", [100, 500, 1000, 2000])
        scheduler_gamma = optimizer_params.get("scheduler_gamma", 0.5)
        scheduler_T_max = optimizer_params.get("scheduler_T_max", 1000)
        scheduler_eta_min = optimizer_params.get("scheduler_eta_min", 0.01)
        scheduler_T_0 = optimizer_params.get("scheduler_T_0", 100)
        scheduler_T_mult = optimizer_params.get("scheduler_T_mult", 3)
        if scheduler == "MultiStepLR":
            self._scheduler = optim.lr_scheduler.MultiStepLR(
                self._optimizer, milestones=scheduler_milestones, gamma=scheduler_gamma)
        elif scheduler == "CosineAnnealingLR":
            self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=scheduler_T_max, eta_min=scheduler_eta_min)
        elif scheduler == "CosineAnnealingWarmRestarts":
            self._scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer, T_0=scheduler_T_0, T_mult=scheduler_T_mult, eta_min=scheduler_eta_min)
        elif scheduler == "Special":
            maxlr = optimizer_params.get("maxlr", 0.1)
            minlr = optimizer_params.get("minlr", 0.01)
            upepoch = optimizer_params.get("upepoch", 10000)
            uprate = (maxlr - minlr) / upepoch
            decayepoch = optimizer_params.get("decayepoch", 20000)
            decayrate = optimizer_params.get("decayrate", 0.95)
            stop_change_epoch = optimizer_params.get("stop_change_epoch", 30000)
            stop_lr = optimizer_params.get("stop_lr", 0.001)
            lambda_lr = lambda epoch: epoch * uprate + minlr if epoch < upepoch else (
                maxlr if epoch < decayepoch else maxlr * (
                        decayrate ** ((epoch - decayepoch) // 200))) if epoch < stop_change_epoch else stop_lr
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lambda_lr)
        elif scheduler == "Constant":
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lambda epoch: 1)
        else:
            raise ValueError("Invalid Scheduler")

    def __train(self):
        """
        train the network
        """
        self._model.train()  # set the model to train mode
        train_loss = 0  # initialize the loss
        data_loader = self._train_dataset.dataloader  # get the data loader
        weight_all = torch.tensor([],dtype=torch.float32,device=self._device)
        x_true = torch.tensor([],dtype=torch.float32,device=self._device)
        y_true = torch.tensor([],dtype=torch.float32,device=self._device)
        y_pred = torch.tensor([],dtype=torch.float32,device=self._device)
        for index, (data, coef, label, data_index) in enumerate(data_loader):
            # move the data to gpu
            data, coef, label = data.to(self._device), coef.to(self._device), label.to(self._device)

            self._optimizer.zero_grad()  # zero the gradient
            if self._optimizer_name == "Adagrad":
                # move optimizer state to gpu
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self._device)

            x_true = torch.cat((x_true, coef), 0)
            y_true = torch.cat((y_true, label), 0)
            weight = self._model(data)

            weight_all = torch.cat((weight_all, weight.to(torch.float32)), 0)
            output = self._out(weight.mul(coef))
            y_pred = torch.cat((y_pred, output), 0)
            loss = self._criterion(output, label) # calculate the loss
            loss.backward()  # back propagation
            self._optimizer.step()  # update the parameters
            if isinstance(data, list):
                train_loss += loss.item() * data[0].size(0)
            else:
                train_loss += loss.item() * data.size(0)  # accumulate the loss

        self._train_diagnosis = DIAGNOSIS(weight_all, x_true, y_true, y_pred)
        train_loss /= self._train_dataset.datasize  # calculate the average loss
        self._trainLossList.append(train_loss)  # record the loss

    def __valid(self):
        """
        validate the network
        """
        self._model.eval()  # set the model to validation mode
        val_loss = 0  # initialize the loss
        label_list = torch.tensor([],dtype=torch.float32,device=self._device)
        out_list = torch.tensor([],dtype=torch.float32,device=self._device)

        data_loader = self._valid_dataset.dataloader  # get the data loader

        with torch.no_grad():  # disable gradient calculation
            for data, coef, label, data_index in data_loader:
                data, coef, label = data.to(self._device), coef.to(self._device), label.to(self._device)
                # weight = self._model(data)
                output = self._out(self._model(
                    data).mul(coef.to(torch.float32)))
                loss = self._criterion(output, label)  # calculate the loss
                out_list = torch.cat((out_list, output.view(-1)))
                label_list = torch.cat((label_list, label.view(-1)))
                if isinstance(data, list):
                    val_loss += loss.item() * data[0].size(0)
                else:
                    val_loss += loss.item() * data.size(0)  # accumulate the loss
            val_loss /= len(self._valid_dataset)  # calculate the average loss
            self._validLossList.append(val_loss)  # record the loss
            try:
                r2 =1 - torch.sum((out_list - label_list) ** 2) / torch.sum((label_list - torch.mean(label_list)) ** 2)
            except:
                if np.isnan(out_list).sum() > 0:
                    raise ValueError("The output contains nan value")
                else:
                    raise ValueError("The Unexpected Error")
            self._valid_r2 = r2
            if r2 > self._bestr2:
                # if the R square is better than the best R square,record the R square and save the model
                self._bestr2 = r2
                self._besttrainr2 = self._train_diagnosis.R2().data
                self._noUpdateEpoch = 0
                if not os.path.exists(self._modelSavePath):
                    os.mkdir(self._modelSavePath)
                torch.save(self._model, self._modelSavePath + '/' + self._modelName + ".pkl")
            else:
                self._noUpdateEpoch += 1

    def __evaluate(self, dataset):
        """
        test the network
        """
        self._model.eval()
        test_loss = 0
        out_list = torch.tensor([],dtype=torch.float32,device=self._device)
        label_list = torch.tensor([],dtype=torch.float32,device=self._device)
        data_loader = dataset.dataloader # dataset
        x_data = torch.tensor([],dtype=torch.float32,device=self._device)
        y_data = torch.tensor([],dtype=torch.float32,device=self._device)
        y_pred = torch.tensor([],dtype=torch.float32,device=self._device)
        weight_all = torch.tensor([],dtype=torch.float32,device=self._device)
        with torch.no_grad():
            for data, coef, label, data_index in data_loader:
                data, coef, label = data.to(self._device), coef.to(self._device), label.to(self._device)
                x_data = torch.cat((x_data, coef), 0)
                y_data = torch.cat((y_data, label), 0)
                weight = self._model(data)
                weight_all = torch.cat((weight_all, weight.to(torch.float32)), 0)
                output = self._out(weight.mul(coef.to(torch.float32)))
                y_pred = torch.cat((y_pred, output), 0)
                loss = self._criterion(output, label)

                out_list = torch.cat((out_list, output.view(-1)))
                label_list = torch.cat((label_list, label.view(-1)))

                if isinstance(data, list):
                    test_loss += loss.item() * data[0].size(0)
                else:
                    test_loss += loss.item() * data.size(0)  # accumulate the loss
            
            test_loss /= len(dataset)

            return test_loss, DIAGNOSIS(weight_all, x_data, y_data, y_pred)

    def run(self, max_epoch=1, early_stop=-1,**kwargs):
        """
        train the model and validate the model

        Parameters
        ----------
        max_epoch : int
            the max epoch of the training (default: ``1``)
        early_stop : int
            if the model has not been updated for ``early_stop`` epochs, the training will stop (default: ``-1``)

            if ``early_stop`` is ``-1``, the training will not stop until the max epoch
        print_frequency : int
            the frequency of printing the information (default: ``50``)

        show_detailed_info : bool
            if ``True``, the detailed information will be shown (default: ``True``)
        """
        if kwargs.get("print_frequency") is not None:
            warnings.warn("The parameter print_frequency is deprecated, the information will be shown in tqdm")
        if kwargs.get("show_detailed_info") is not None:
            warnings.warn("The parameter show_detailed_info is deprecated, the information will be shown in tqdm")
        # model selection method
        model_selection = kwargs.get("model_selection", "val")
        self.__istrained = True
        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)  # parallel computing
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        # create file
        if not os.path.exists(self._log_path):
            os.mkdir(self._log_path)
        file_str = self._log_path + self._log_file_name
        logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            filename=file_str, level=logging.INFO)
        with tqdm(range(max_epoch)) as pbar:
            for epoch in pbar:
                self._epoch = epoch
                # train the network
                # record the information of the training process
                self.__train()
                # validate the network
                # record the information of the validation process
                self.__valid()
                # out put the information
                pbar.set_postfix({'Train Loss': "{:5f}".format(self._trainLossList[-1]), 'Train R2': "{:5f}".format(self._train_diagnosis.R2().data.cpu().numpy()),
                                  'Train AIC': self._train_diagnosis.AIC(),'Valid Loss': self._validLossList[-1],
                                  'Valid R2': self._valid_r2.item(), 'Best Valid R2': self._bestr2.item(),
                                  'Learning Rate': self._optimizer.param_groups[0]['lr']})

                self._scheduler.step()  # update the learning rate
                # tensorboard
                self._writer.add_scalar('Training/Learning Rate', self._optimizer.param_groups[0]['lr'], self._epoch)
                self._writer.add_scalar('Training/Loss', self._trainLossList[-1], self._epoch)
                self._writer.add_scalar('Training/R2', self._train_diagnosis.R2().data, self._epoch)
                self._writer.add_scalar('Training/RMSE', self._train_diagnosis.RMSE().data, self._epoch)
                self._writer.add_scalar('Training/AIC', self._train_diagnosis.AIC(), self._epoch)
                self._writer.add_scalar('Training/AICc', self._train_diagnosis.AICc(), self._epoch)
                self._writer.add_scalar('Validation/Loss', self._validLossList[-1], self._epoch)
                self._writer.add_scalar('Validation/R2', self._valid_r2, self._epoch)
                self._writer.add_scalar('Validation/Best R2', self._bestr2, self._epoch)

                # log output
                log_str = "Epoch: " + str(epoch + 1) + \
                          "; Train Loss: " + str(self._trainLossList[-1]) + \
                          "; Train R2: {:5f}".format(self._train_diagnosis.R2().data) + \
                          "; Train RMSE: {:5f}".format(self._train_diagnosis.RMSE().data) + \
                          "; Train AIC: {:5f}".format(self._train_diagnosis.AIC()) + \
                          "; Train AICc: {:5f}".format(self._train_diagnosis.AICc()) + \
                          "; Valid Loss: " + str(self._validLossList[-1]) + \
                          "; Valid R2: " + str(self._valid_r2) + \
                          "; Learning Rate: " + str(self._optimizer.param_groups[0]['lr'])
                logging.info(log_str)
                if 0 < early_stop < self._noUpdateEpoch:  # stop when the model has not been updated for long time
                    print("Training stop! Model has not been improved for over {} epochs.".format(early_stop))
                    break
        torch.save(self._model, self._modelSavePath + '/' + self._modelName + "_last.pkl")
        if model_selection == "val":
            self.load_model(self._modelSavePath + '/' + self._modelName + ".pkl")
        elif model_selection == "last":
            self.load_model(self._modelSavePath + '/' + self._modelName + "_last.pkl")
        self.result_data = self.getCoefs()

    def predict(self, dataset):
        """
        predict the result of the dataset

        Parameters
        ----------
        dataset : baseDataset,predictDataset
            the dataset to be predicted

        Returns
        -------
        dataframe
            the Pandas dataframe of the dataset with the predicted result
        """
        
        data_loader = dataset.dataloader
        dataset.y_scale_info = self._model_y_scale_info
        if not self.__istrained:
            print("WARNING! The model hasn't been trained or loaded!")
        self._model.eval()
        result = torch.tensor([]).to(torch.float32)
        with torch.no_grad():
            for batch in data_loader:
                data = batch[0]
                coef = batch[1]
                if self._use_gpu:
                    result, data, coef = result.cuda(), data.cuda(), coef.cuda()
                    ols_w = torch.tensor(self._coefficient).to(torch.float32).cuda()
                else:
                    ols_w = torch.tensor(self._coefficient).to(torch.float32)
                weight = self._model(data)
                result = torch.cat((result, self._out(weight.mul(coef))), 0)
        result = result.cpu().detach().numpy()
        dataset.dataframe['pred_result'] = result
        if dataset.y_scale_info:
            _,dataset.dataframe['denormalized_pred_result'] = dataset.rescale(None,result)
        else:
            dataset.dataframe['denormalized_pred_result'] = result
        dataset.pred_result = result
        return dataset.dataframe

    def predict_coef(self, dataset):
        """
        predict the spatial coefficient of the independent variable

        Parameters
        ----------
        dataset : baseDataset,predictDataset
            the dataset to be predicted

        Returns
        -------
        dataframe
            the Pandas dataframe of the dataset with the predicted spatial coefficient
        """
        data_loader = dataset.dataloader
        if not self.__istrained:
            print("WARNING! The model hasn't been trained or loaded!")
        self._model.eval()
        result = torch.tensor([]).to(torch.float32)
        with torch.no_grad():
            for batch in data_loader:
                data = batch[0]
                coef = batch[1]
                if self._use_gpu:
                    result, data, coef = result.cuda(), data.cuda(), coef.cuda()
                    ols_w = torch.tensor(self._coefficient).to(torch.float32).cuda()
                else:
                    ols_w = torch.tensor(self._coefficient).to(torch.float32)
                coefficient = self._model(data).mul(ols_w)
                result = torch.cat((result, coefficient), 0)
        result = result.cpu().detach().numpy()
        return result

    def load_model(self, path, use_dict=False, map_location=None):
        """
        load the model from the path

        Parameters
        ----------
        path : str
            the path of the model
        use_dict : bool
            whether the function use dict to load the model (default: ``False``)
        map_location : str
            the location of the model (default: ``None``)
            the location can be ``"cpu"`` or ``"cuda"``
        """
        if use_dict:
            data = torch.load(path, map_location=map_location, weights_only=False)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path, map_location=map_location, weights_only=False)
        if self._use_gpu:
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        else:
            self._model = self._model.cpu()
            self._out = self._out.cpu()
        self._modelSavePath = os.path.dirname(path)
        self._modelName = os.path.basename(path).split('/')[-1].split('.')[0]
        self.__istrained = True
        self.result_data = self.getCoefs()

    def gpumodel_to_cpu(self, path, save_path, use_model=True):
        """
        convert gpu model to cpu model

        Parameters
        ----------
        path : str
            the path of the model
        save_path : str
            the path of the new model
        use_model : bool
            whether use dict to load the model (default: ``True``)
        """
        if use_model:
            data = torch.load(path, map_location='cpu', weights_only=False).state_dict()
        else:
            data = torch.load(path, map_location='cpu', weights_only=False)
        new_state_dict = OrderedDict()
        for k, v in data.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        torch.save(new_state_dict, save_path)

    def getLoss(self):
        """
        get network's loss

        Returns
        -------
        list
            the list of the loss in training process and validation process
        """
        return self._trainLossList, self._validLossList

    def add_graph(self):
        """
        add the graph of the model to tensorboard
        """
        for data, coef, label, data_index in self._train_dataset.dataloader:
            if self._use_gpu:
                data = data.cuda()
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()
                data = data.cpu()
            self._writer.add_graph(self._model, data)
            break
        print("Add Graph Successfully")

    def result(self, path=None, use_dict=False, map_location=None):
        """
        print the result of the model, including the model name, regression fomula and the result of test dataset

        Parameters
        ----------
        path : str
            the path of the model(default: ``None``)
            | if ``path`` is ``None``, the model will be loaded from ``self._modelSavePath + "/" + self._modelName + ".pkl"``
        use_dict : bool
            whether the function use dict to load the model (default: ``False``)
            | if ``use_dict`` is ``True``, the model will be loaded from ``path`` as dict
        map_location : str
            the location of the model (default: ``None``)
            the location can be ``"cpu"`` or ``"cuda"``
        """
        # load model
        if not self.__istrained:
            raise Exception("The model hasn't been trained or loaded!")
        if path is None:
            path = self._modelSavePath + "/" + self._modelName + ".pkl"
        if use_dict:
            data = torch.load(path, map_location=map_location, weights_only=False)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(path, map_location=map_location, weights_only=False)
        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)  # parallel computing
            self._model = self._model.cuda()
            self._out = self._out.cuda()
        else:
            self._model = self._model.cpu()
            self._out = self._out.cpu()
        with torch.no_grad():
            _ , self._train_diagnosis = self.__evaluate(self._train_dataset)
            self._trainr2 = self._train_diagnosis.R2().data
            _ , self._valid_diagnosis = self.__evaluate(self._valid_dataset)
            self._validr2 = self._valid_diagnosis.R2().data
            self.__testLoss, self._test_diagnosis = self.__evaluate(self._test_dataset)
            self.__testr2 = self._test_diagnosis.R2().data


        logging.info("Test Loss: " + str(self.__testLoss) + "; Test R2: " + str(self.__testr2))
        # print result
        # basic information
        print("--------------------Model Information-----------------")
        print("Model Name:           |", self._modelName)
        print("independent variable: |", self._train_dataset.x)
        print("dependent variable:   |", self._train_dataset.y)
        # OLS
        print("\nOLS coefficients: ")
        for i in range(len(self._coefficient)):
            if i == len(self._coefficient) - 1:
                print("Intercept: {:.5f}".format(self._coefficient[i]))
            else:
                print("x{}: {:.5f}".format(i, self._coefficient[i]))
        print("\n--------------------Result Information----------------")
        print("Test Loss: | {:>25.5f}".format(self.__testLoss))
        print("Test R2  : | {:>25.5f}".format(self.__testr2))
        print("Train R2 : | {:>25.5f}".format(self._trainr2))
        print("Valid R2 : | {:>25.5f}".format(self._validr2))
        print("RMSE: | {:>30.5f}".format(self._test_diagnosis.RMSE().data))
        print("AIC:  | {:>30.5f}".format(self._test_diagnosis.AIC()))
        print("AICc: | {:>30.5f}".format(self._test_diagnosis.AICc()))
        print("F1:   | {:>30.5f}".format(self._test_diagnosis.F1_Global().data))
        print("F2:   | {:>30.5f}".format(self._test_diagnosis.F2_Global().flatten()[0].data))
        F3_Local_dict = self._test_diagnosis.F3_Local()[0]
        for key in F3_Local_dict:
            width = 30 - (len(key) - 4)
            print("{}: | {:>{width}.5f}".format(key, F3_Local_dict[key].data, width=width))

    def reg_result(self, filename=None, model_path=None, use_dict=False, only_return=False, map_location=None):
        """
        save the regression result of the model, including the coefficient of each argument, the bias and the predicted result

        Parameters
        ----------
        filename : str
            the path of the result file (default: ``None``)
            | if ``filename`` is ``None``, the result will not be saved as file
        model_path : str
            the path of the model (default: ``None``)
            | if ``model_path`` is ``None``, the model will be loaded from ``self._modelSavePath + "/" + self._modelName + ".pkl"``
        use_dict : bool
            whether use dict to load the model (default: ``False``)
            | if ``use_dict`` is ``True``, the model will be loaded from ``model_path`` as dict
        only_return : bool
            whether only return the result (default: ``False``)
            | if ``only_return`` is ``True``, the result will not be saved as file
        map_location : str
            the location of the model (default: ``None``)
            the location can be ``"cpu"`` or ``"cuda"``

        Returns
        -------
        dataframe
            the Pandas dataframe of the result
        """
        if model_path is None:
            model_path = self._modelSavePath + "/" + self._modelName + ".pkl"
            
        if use_dict:
            data = torch.load(model_path, map_location=map_location, weights_only=False)
            self._model.load_state_dict(data)
        else:
            self._model = torch.load(model_path, map_location=map_location, weights_only=False)

        if self._use_gpu:
            self._model = nn.DataParallel(module=self._model)
            self._model,self._out = self._model.cuda(),self._out.cuda()
        else:
            self._model, self._out = self._model.cpu(), self._out.cpu()

        device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
        result = torch.tensor([]).to(torch.float32).to(device)
        train_data_size = valid_data_size = 0
        
        with torch.no_grad():
            # calculate the result of train dataset
            for data, coef, label, data_index in self._train_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(
                    device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)
            train_data_size = len(result)
            # calculate the result of train dataset
            for data, coef, label, data_index in self._valid_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(
                    device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)
            valid_data_size = len(result) - train_data_size
            # calculate the result of train dataset
            for data, coef, label, data_index in self._test_dataset.dataloader:
                data, coef, label, data_index = data.to(device), coef.to(device), label.to(device), data_index.to(
                    device)
                output = self._out(self._model(data).mul(coef.to(torch.float32)))
                coefficient = self._model(data).mul(torch.tensor(self._coefficient).to(torch.float32).to(device))
                output = torch.cat((coefficient, output, data_index), dim=1)
                result = torch.cat((result, output), 0)

        result = result.cpu().detach().numpy()
        columns = list(self._train_dataset.x)
        for i in range(len(columns)):
            columns[i] = "coef_" + columns[i]
        columns.append("bias")
        columns = columns + ["Pred_" + self._train_dataset.y[0]] + self._train_dataset.id
        result = pd.DataFrame(result, columns=columns)
        result[self._train_dataset.id] = result[self._train_dataset.id].astype(np.int32)
        result["Pred_" + self._train_dataset.y[0]] = result["Pred_" + self._train_dataset.y[0]].astype(np.float32)

        # set dataset belong to postprocess
        result['dataset_belong'] = np.concatenate([
            np.full(train_data_size, 'train'),
            np.full(valid_data_size, 'valid'),
            np.full(len(result) - train_data_size - valid_data_size, 'test')
        ])

        # denormalize pred result
        if self._train_dataset.y_scale_info:
            _, result['denormalized_pred_result'] = self._train_dataset.rescale(None,result)
        else:
            result['denormalized_pred_result'] = result["Pred_" + self._train_dataset.y[0]]
        
        if only_return:
            return result
        
        if filename is not None:
            result.to_csv(filename, index=False)
        else:
            warnings.warn(
                "Warning! The input write file path is not set. Result is returned by function but not saved as file.",
                RuntimeWarning)
        return result

    def getCoefs(self):
        """
        get the Coefficients of each argument in dataset

        Returns
        -------
        dataframe
            the Pandas dataframe of the coefficient of each argument in dataset
        """
        result_data = self.reg_result(only_return=True)
        result_data['id'] = result_data['id'].astype(np.int64)
        data = pd.concat([self._train_dataset.dataframe, self._valid_dataset.dataframe, self._test_dataset.dataframe])
        data.set_index('id', inplace=True)
        result_data.set_index('id', inplace=True)
        result_data = result_data.join(data)
        return result_data

    def __str__(self) -> str:
        print("Model Name: ", self._modelName)
        print("Model Structure: ", self._model)
        return ""

    def __repr__(self) -> str:
        print("Model Name: ", self._modelName)
        print("Model Structure: ", self._model)
        return ""


class GTNNWR(GNNWR):
    """
    GTNNWR model is a model based on GNNWR and STPNN, which is a model that can be used to solve the problem of
    spatial-temporal non-stationarity.

    Parameters
    ----------
    train_dataset : baseDataset
        the dataset for training
    valid_dataset : baseDataset
        the dataset for validation
    test_dataset : baseDataset
        the dataset for test
    dense_layers : list
        the dense layers of the model (default: ``None``)
        | i.e. ``[[3],[128,64,32]]`` the first list in input is hidden layers of STPNN, the second one is hidden layers of SWNN.
    start_lr : float
        the start learning rate (default: ``0.1``)
    optimizer : str, optional
        the optimizer of the model (default: ``"Adagrad"``)
        choose from "SGD","Adam","RMSprop","Adagrad","Adadelta"
    drop_out : float
        the drop out rate of the model (default: ``0.2``)
    batch_norm : bool, optional
        whether use batch normalization (default: ``True``)
    activate_func : torch.nn
        the activate function of the model (default: ``nn.PReLU(init=0.4)``)
    model_name : str
        the name of the model (default: ``"GNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")``)
    model_save_path : str
        the path of the model (default: ``"../gnnwr_models"``)
    write_path : str
        the path of the log (default: ``"../gnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")``)
    use_gpu : bool
        whether use gpu or not (default: ``True``)
    use_ols : bool
        whether use ols or not (default: ``True``)
    log_path : str
        the path of the log (default: ``"../gnnwr_logs/"``)
    log_file_name : str
        the name of the log (default: ``"gnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log"``)
    log_level : int
        the level of the log (default: ``logging.INFO``)
    optimizer_params : dict, optional
        the params of the optimizer and the scheduler (default: ``None``)

        if optimizer is SGD, the params are:

            | maxlr: float, the max learning rate (default: ``0.1``)

            | minlr: float, the min learning rate (default: ``0.01``)

            | upepoch: int, the epoch of learning rate up (default: ``10000``)

            | decayepoch: int, the epoch of learning rate decay (default: ``20000``)

            | decayrate: float, the rate of learning rate decay (default: ``0.1``)

            | stop_change_epoch: int, the epoch of learning rate stop change (default: ``30000``)

            | stop_lr: float, the learning rate when stop change (default: ``0.001``)

        if optimizer is Other, the params are:

            | scheduler: str, the name of the scheduler (default: ``"CosineAnnealingWarmRestarts"``) in {``"MultiStepLR","CosineAnnealingLR","CosineAnnealingWarmRestarts"``}

            | scheduler_milestones: list, the milestones of the scheduler MultiStepLR (default: ``[500,1000,2000,4000]``)

            | scheduler_gamma: float, the gamma of the scheduler MultiStepLR (default: ``0.5``)

            | scheduler_T_max: int, the T_max of the scheduler CosineAnnealingLR (default: ``1000``)

            | scheduler_eta_min: float, the eta_min of the scheduler CosineAnnealingLR and CosineAnnealingWarmRestarts (default: ``0.01``)

            | scheduler_T_0: int, the T_0 of the scheduler CosineAnnealingWarmRestarts (default: ``100``)

            | scheduler_T_mult: int, the T_mult of the scheduler CosineAnnealingWarmRestarts (default: ``3``)
    STPNN_outsize:int
        the output size of STPNN(default:``1``)
    STNN_SPNN_params:dict
        the params of STNN and SPNN(default:``None``)

        STPNN_batch_norm:bool

            whether use batchnorm in STNN and SPNN or not (Default:``True``)
    """

    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 dense_layers=None,
                 start_lr: float = .1,
                 optimizer="Adam",
                 drop_out=0.2,
                 batch_norm=True,
                 activate_func=nn.PReLU(init=0.4),
                 model_name="GTNNWR_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
                 model_save_path="../gtnnwr_models",
                 write_path="../gtnnwr_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                 use_gpu: bool = True,
                 use_ols: bool = True,
                 log_path: str = "../gtnnwr_logs/",
                 log_file_name: str = "gtnnwr" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
                 log_level: int = logging.INFO,
                 optimizer_params=None,
                 STPNN_outsize=1,
                 STNN_SPNN_params=None,
                 ):

        if dense_layers is None:
            dense_layers = [[], []]
        super(GTNNWR, self).__init__(train_dataset, valid_dataset, test_dataset, dense_layers[1], start_lr, optimizer,
                                     drop_out, batch_norm, activate_func, model_name, model_save_path, write_path,
                                     use_gpu, use_ols, log_path, log_file_name, log_level, optimizer_params)
        self._STPNN_out = STPNN_outsize
        self._modelName = model_name  # model name
        if train_dataset.simple_distance:
            insize = 2
        else:
            insize = train_dataset.distances.shape[-1]
        if STNN_SPNN_params is None:
            STNN_SPNN_params = dict()
        self.STNN_outsize = STNN_SPNN_params.get("STNN_outsize", 1)
        self.SPNN_outsize = STNN_SPNN_params.get("SPNN_outsize", 1)
        self.STPNN_batch_norm = STNN_SPNN_params.get("STPNN_batch_norm", True)
        if train_dataset.is_need_STNN:
            self._model = nn.Sequential(STNN_SPNN(train_dataset.temporal.shape[-1], self.STNN_outsize,
                                                  train_dataset.distances.shape[-1], self.SPNN_outsize),
                                        STPNN(dense_layers[0], self.STNN_outsize + self.SPNN_outsize,
                                              self._STPNN_out, drop_out, batch_norm=self.STPNN_batch_norm),
                                        SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                             activate_func, batch_norm))
        else:
            self._model = nn.Sequential(STPNN(dense_layers[0], insize, self._STPNN_out, drop_out,
                                              batch_norm=self.STPNN_batch_norm),
                                        SWNN(dense_layers[1], self._STPNN_out * self._insize, self._outsize, drop_out,
                                             activate_func, batch_norm))
        self.init_optimizer(optimizer, optimizer_params)


# 損失関数の負の対数尤度損失関数
# def NegativeLogLikelihoodLoss(predicted_density_map, accident_arixel_indices):
#     """
#     負の対数尤度損失関数

#     Parameters
#     ----------
#     predicted_density_map : torch.Tensor
#         モデルが出力した、領域全体の全てのarixelのリスク密度
#         (例: [0.0001, 0.0005, 0.0002, ...])

#     accident_arixel_indices : torch.Tensor
#         実際に事故が起きたarixelのインデックス（番号）のリスト
#         (例: [1, 5, 87, 204, ...])

#     Returns
#     -------
#     torch.Tensor
#         計算された損失（スカラー値）
#     """
    
#     # 1. 密度マップの中から、実際に事故が起きた地点の密度だけを抜き出す
#     densities_at_accidents = predicted_density_map[accident_arixel_indices]

#     # 2. 対数を計算する (非常に小さい値が含まれる可能性があるので、微小な値を足して安定させる)
#     log_likelihoods = torch.log(densities_at_accidents + 1e-10) # 1e-10はゼロ除算を防ぐため

#     # 3. 全ての対数尤度を合計し、マイナスをつけて損失とする
#     loss = -torch.sum(log_likelihoods)
    
#     return loss

class NN_STNKDE:
    r"""
    NN_STNKDE (Neural Network Spatio-Temporal Kernel Density Estimation) is a model that learns
    spatio-temporal kernel weights using neural networks, similar to STNKDE but with learned kernels.
    
    The model consists of:
    1. STPNN: Learns spatio-temporal distance representation from accident arixels
    2. STWNN: Outputs kernel weights from learned distances (non-negative, normalized to sum=1)

    Parameters
    ----------
    train_dataset : STNKDE_Dataset
        Dataset containing accident arixels for training
    valid_dataset : STNKDE_Dataset
        Dataset for validation
    test_dataset : STNKDE_Dataset
        Dataset for testing
    dense_layers : list
        Dense layer structure: [[STPNN_layers], [STWNN_layers]]
        e.g., [[64, 32], [128, 64]] means STPNN has [64, 32] and STWNN has [128, 64]
    start_lr : float
        Initial learning rate (default: 0.001)
    optimizer : str
        Optimizer name: "SGD", "Adam", "RMSprop", "Adagrad", "Adadelta" (default: "Adam")
    drop_out : float
        Dropout rate (default: 0.2)
    batch_norm : bool
        Whether to use batch normalization (default: True)
    activate_func : torch.nn
        Activation function (default: nn.PReLU(init=0.4))
    model_name : str
        Model name (default: auto-generated)
    model_save_path : str
        Path to save models (default: "../nn_stnkde_models")
    write_path : str
        Path for TensorBoard logs (default: auto-generated)
    use_gpu : bool
        Whether to use GPU (default: True)
    log_path : str
        Path for log files (default: "../nn_stnkde_logs")
    log_file_name : str
        Log file name (default: auto-generated)
    log_level : int
        Logging level (default: logging.INFO)
    optimizer_params : dict
        Parameters for optimizer and scheduler
    STPNN_outsize : int
        Output size of STPNN (default: 1)
    use_softmax : bool
        Whether to use softmax for normalization (default: True)
        If False, uses sum normalization
    density_loss_normalization : {"none","l1","softmax"}
        Strategy for normalizing per-sample densities during training (default: "l1").
    density_softmax_temperature : float
        Temperature applied when ``density_loss_normalization="softmax"``.
    use_soft_cutoff : bool
        Whether to learn a soft cutoff gate over distances.
    soft_cutoff_temperature : float
        Temperature controlling softness of the learned cutoff.
    hard_cutoff_distance : float, optional
        Optional deterministic cutoff radius applied to pairwise distances (on normalized scale).
    learn_hard_cutoff : bool
        If True, treat the hard cutoff radius as a learnable parameter.
    stwnn_chunk_size : int
        Process STWNN inputs in chunks of this size to reduce memory (0 disables chunking).
    """
    
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 dense_layers=None,
                 start_lr: float = 0.001,
                 optimizer="Adam",
                 drop_out=0.2,
                 batch_norm=True,
                 activate_func=nn.PReLU(init=0.4),
                 model_name="NN_STNKDE_" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S"),
                 model_save_path="../nn_stnkde_models",
                 write_path="../nn_stnkde_runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                 use_gpu: bool = True,
                 log_path: str = "../nn_stnkde_logs/",
                 log_file_name: str = "nn_stnkde" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log",
                 log_level: int = logging.INFO,
                 optimizer_params=None,
                 STPNN_outsize=1,
                 use_softmax=True,
                 density_loss_normalization="l1",
                 density_softmax_temperature=1.0,
                 use_soft_cutoff=True,
                 soft_cutoff_temperature=1.0,
                 hard_cutoff_distance=None,
                 learn_hard_cutoff=False,
                 stwnn_chunk_size=0):
        
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._use_softmax = use_softmax
        self._density_loss_norm = (density_loss_normalization or "none").lower()
        self._density_softmax_temp = density_softmax_temperature
        self._use_soft_cutoff = use_soft_cutoff
        self._soft_cutoff_temperature = soft_cutoff_temperature
        self._stwnn_chunk_size = stwnn_chunk_size
        self._context_dim = getattr(train_dataset, "context_dim", 0)
        
        # Get input/output sizes
        # STPNN input: 2 (spatial_dist, temporal_dist)
        stpnn_insize = 2
        self._STPNN_outsize = STPNN_outsize
        
        # STWNN input: STPNN_outsize (learned distance representation)
        # STWNN output: 1 (weight for each reference arixel)
        if dense_layers is None:
            dense_layers = [[], []]
        stpnn_layers = dense_layers[0] if len(dense_layers) > 0 else []
        stwnn_layers = dense_layers[1] if len(dense_layers) > 1 else []
        
        # Number of reference arixels retained after optional neighbor pruning
        self._n_reference = train_dataset.distances.shape[1]
        # STWNN input includes: STPNN output + reference context + target context
        # This allows learning interactions like "accident at intersection affects another intersection differently"
        stwnn_insize = self._STPNN_outsize + 2 * self._context_dim
        stwnn_outsize = 1  # Output one weight per reference arixel
        
        # Build model: STPNN -> STWNN
        # STPNN: (batch, n_reference, 2) -> (batch, n_reference * STPNN_outsize)
        # We need to reshape and apply SWNN to each reference point
        self._stpnn = STPNN(stpnn_layers, stpnn_insize, self._STPNN_outsize, drop_out, 
                           activate_func=nn.ReLU(), batch_norm=batch_norm)
        self._stwnn = SWNN(stwnn_layers, stwnn_insize, stwnn_outsize, drop_out, 
                          activate_func=activate_func, batch_norm=batch_norm)
        self._soft_cutoff_layer = None
        if self._use_soft_cutoff:
            self._soft_cutoff_layer = nn.Linear(stwnn_insize, 1)
        self._hard_cutoff_param = None
        self._learn_hard_cutoff = learn_hard_cutoff
        if hard_cutoff_distance is not None:
            base_value = float(hard_cutoff_distance)
            if learn_hard_cutoff:
                self._hard_cutoff_param = nn.Parameter(torch.tensor(base_value, dtype=torch.float32))
            else:
                self._hard_cutoff_param = base_value
        
        # Loss function: Negative Log Likelihood
        # We maximize the density at arixels where accidents actually occurred
        # Loss = -log(density) for accident arixels
        # This will be computed manually in the training loop
        
        # Training state
        self._trainLossList = []
        self._validLossList = []
        self._epoch = 0
        self._best_loss = float('inf')
        self._noUpdateEpoch = 0
        self._modelName = model_name
        self._modelSavePath = model_save_path
        self.__istrained = False
        
        # GPU setup
        self._use_gpu = use_gpu
        if self._use_gpu:
            if torch.cuda.is_available():
                devices = [i for i in range(torch.cuda.device_count())]
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
            else:
                self._use_gpu = False
        
        self._device = torch.device('cuda') if self._use_gpu else torch.device('cpu')
        
        # Optimizer setup
        self._optimizer = None
        self._scheduler = None
        self._optimizer_name = None
        self._start_lr = start_lr
        self.init_optimizer(optimizer, optimizer_params)
        
        # Logging setup
        self._log_path = log_path
        self._log_file_name = log_file_name
        self._log_level = log_level
        self._writer = SummaryWriter(write_path)
        
        # Results storage
        self.result_data = None
    
    def init_optimizer(self, optimizer, optimizer_params=None):
        """Initialize optimizer and scheduler."""
        if optimizer_params is None:
            optimizer_params = {}
        
        self._optimizer_name = optimizer
        
        # Combine parameters from both networks
        params = list(self._stpnn.parameters()) + list(self._stwnn.parameters())
        if self._soft_cutoff_layer is not None:
            params += list(self._soft_cutoff_layer.parameters())
        if isinstance(self._hard_cutoff_param, nn.Parameter):
            params.append(self._hard_cutoff_param)
        if optimizer == "SGD":
            self._optimizer = optim.SGD(params, lr=self._start_lr,
                                       weight_decay=optimizer_params.get("weight_decay", 0))
        elif optimizer == "Adam":
            self._optimizer = optim.Adam(params, lr=self._start_lr,
                                        weight_decay=optimizer_params.get("weight_decay", 0))
        elif optimizer == "RMSprop":
            self._optimizer = optim.RMSprop(params, lr=self._start_lr,
                                           weight_decay=optimizer_params.get("weight_decay", 0))
        elif optimizer == "Adagrad":
            self._optimizer = optim.Adagrad(params, lr=self._start_lr,
                                          weight_decay=optimizer_params.get("weight_decay", 0))
        elif optimizer == "Adadelta":
            self._optimizer = optim.Adadelta(params, lr=self._start_lr,
                                            weight_decay=optimizer_params.get("weight_decay", 0))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # Scheduler setup
        scheduler = optimizer_params.get("scheduler", "CosineAnnealingWarmRestarts")
        if scheduler == "MultiStepLR":
            milestones = optimizer_params.get("scheduler_milestones", [500, 1000, 2000, 4000])
            gamma = optimizer_params.get("scheduler_gamma", 0.5)
            self._scheduler = optim.lr_scheduler.MultiStepLR(
                self._optimizer, milestones=milestones, gamma=gamma)
        elif scheduler == "CosineAnnealingLR":
            T_max = optimizer_params.get("scheduler_T_max", 1000)
            eta_min = optimizer_params.get("scheduler_eta_min", 0.01)
            self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler == "CosineAnnealingWarmRestarts":
            T_0 = optimizer_params.get("scheduler_T_0", 100)
            T_mult = optimizer_params.get("scheduler_T_mult", 3)
            eta_min = optimizer_params.get("scheduler_eta_min", 0.01)
            self._scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self._optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        else:
            self._scheduler = optim.lr_scheduler.LambdaLR(
                self._optimizer, lr_lambda=lambda epoch: 1)
    
    def _apply_stwnn(self, stpnn_flat):
        """Apply STWNN with optional chunking to save memory."""
        if self._stwnn_chunk_size and self._stwnn_chunk_size > 0:
            outputs = []
            for chunk in torch.split(stpnn_flat, self._stwnn_chunk_size, dim=0):
                outputs.append(self._stwnn(chunk))
            return torch.cat(outputs, dim=0)
        return self._stwnn(stpnn_flat)
    
    def _get_hard_cutoff_value(self):
        if isinstance(self._hard_cutoff_param, nn.Parameter):
            return torch.relu(self._hard_cutoff_param)
        return self._hard_cutoff_param
    
    def _apply_cutoffs(self, weights, stpnn_flat, distances, batch_size):
        """Apply hard/soft cutoffs to learned weights."""
        if self._use_soft_cutoff and self._soft_cutoff_layer is not None:
            temp = max(self._soft_cutoff_temperature, 1e-6)
            gating_logits = self._soft_cutoff_layer(stpnn_flat) / temp
            gating = torch.sigmoid(gating_logits).view(batch_size, self._n_reference)
            weights = weights * gating
        hard_value = self._get_hard_cutoff_value()
        if hard_value is not None:
            pairwise_norm = torch.norm(distances, dim=2)
            if not torch.is_tensor(hard_value):
                hard_value = torch.tensor(hard_value, dtype=pairwise_norm.dtype, device=pairwise_norm.device)
            mask = (pairwise_norm <= hard_value).float()
            weights = weights * mask
        return weights
    
    def _compute_pair_weights(self, distances, target_context, reference_context):
        """
        Forward pass for pairwise weights given distances and contextual features.
        
        Parameters
        ----------
        distances : torch.Tensor
            Spatio-temporal distances (batch, n_reference, 2)
        target_context : torch.Tensor or None
            Contextual features of target arixels (batch, context_dim)
        reference_context : torch.Tensor or None
            Contextual features of reference arixels (batch, n_reference, context_dim)
        
        Returns
        -------
        weights : torch.Tensor
            Learned weights (batch, n_reference)
        """
        batch_size = distances.shape[0]
        stpnn_out = self._stpnn(distances)
        stpnn_out = stpnn_out.view(batch_size, self._n_reference, self._STPNN_outsize)
        
        if self._context_dim > 0:
            # Combine STPNN output with both reference and target context features
            # This allows the model to learn interactions like:
            # "Accident at intersection (ref) affects another intersection (target) differently than general road"
            context_parts = []
            
            # Reference context (one per reference point)
            if reference_context is not None:
                context_parts.append(reference_context)  # (batch, n_reference, context_dim)
            else:
                # Zero padding if not provided
                context_parts.append(torch.zeros(batch_size, self._n_reference, self._context_dim, 
                                                dtype=stpnn_out.dtype, device=stpnn_out.device))
            
            # Target context (same for all reference points, but allows learning target-specific effects)
            if target_context is not None:
                target_expanded = target_context.unsqueeze(1).expand(-1, self._n_reference, -1)
                context_parts.append(target_expanded)  # (batch, n_reference, context_dim)
            else:
                # Zero padding if not provided
                context_parts.append(torch.zeros(batch_size, self._n_reference, self._context_dim,
                                                dtype=stpnn_out.dtype, device=stpnn_out.device))
            
            combined_context = torch.cat(context_parts, dim=2)  # (batch, n_reference, 2*context_dim)
            stpnn_out = torch.cat([stpnn_out, combined_context], dim=2)
        else:
            # No context features
            pass
        
        stpnn_flat = stpnn_out.view(batch_size * self._n_reference, self._STPNN_outsize + 2 * self._context_dim)
        weights_raw = self._apply_stwnn(stpnn_flat)
        weights_raw = weights_raw.view(batch_size, self._n_reference, 1).squeeze(-1)
        weights = nn.functional.softplus(weights_raw)
        weights = self._apply_cutoffs(weights, stpnn_flat, distances, batch_size)
        return weights
    
    def _normalize_density(self, density_raw):
        """Normalize density for stable learning."""
        eps = 1e-10
        if self._density_loss_norm == "l1":
            return density_raw / (density_raw.sum() + eps) * density_raw.shape[0]
        if self._density_loss_norm == "softmax":
            temp = max(self._density_softmax_temp, 1e-6)
            logits = torch.log(density_raw + eps) / temp
            probs = torch.softmax(logits, dim=0)
            return probs * density_raw.shape[0]
        return density_raw
    
    @staticmethod
    def _weighted_negative_log_likelihood(density, accident_counts, eps=1e-10):
        """Accident-weighted NLL."""
        weighted_log = accident_counts * torch.log(density + eps)
        return -weighted_log.sum() / (accident_counts.sum() + eps)
    
    def __train(self):
        """Train the model."""
        self._stpnn.train()
        self._stwnn.train()
        train_loss = 0
        data_loader = self._train_dataset.dataloader
        
        for batch in data_loader:
            if len(batch) == 3:
                distances, arixel_ids, accident_counts = batch
                target_context = None
                reference_context = None
            elif len(batch) == 4:
                distances, arixel_ids, accident_counts, target_context = batch
                reference_context = None
            else:
                distances, arixel_ids, accident_counts, target_context, reference_context = batch
            distances = distances.to(self._device)  # (batch, n_reference, 2)
            accident_counts = accident_counts.to(self._device)  # (batch,)
            if self._context_dim > 0:
                if target_context is not None:
                    target_context = target_context.to(self._device)
                if reference_context is not None:
                    reference_context = reference_context.to(self._device)
            else:
                target_context = None
                reference_context = None
            
            self._optimizer.zero_grad()
            
            weights = self._compute_pair_weights(distances, target_context, reference_context)
            density_raw = weights.sum(dim=1)
            density = self._normalize_density(density_raw)
            loss = self._weighted_negative_log_likelihood(density, accident_counts)
            
            loss.backward()
            self._optimizer.step()
            
            train_loss += loss.item() * distances.size(0)
        
        train_loss /= len(self._train_dataset)
        self._trainLossList.append(train_loss)
    
    def __valid(self):
        """Validate the model."""
        self._stpnn.eval()
        self._stwnn.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in self._valid_dataset.dataloader:
                if len(batch) == 3:
                    distances, arixel_ids, accident_counts = batch
                    target_context = None
                    reference_context = None
                elif len(batch) == 4:
                    distances, arixel_ids, accident_counts, target_context = batch
                    reference_context = None
                else:
                    distances, arixel_ids, accident_counts, target_context, reference_context = batch
                distances = distances.to(self._device)
                accident_counts = accident_counts.to(self._device)
                if self._context_dim > 0:
                    if target_context is not None:
                        target_context = target_context.to(self._device)
                    if reference_context is not None:
                        reference_context = reference_context.to(self._device)
                else:
                    target_context = None
                    reference_context = None
                
                weights = self._compute_pair_weights(distances, target_context, reference_context)
                density_raw = weights.sum(dim=1)
                density = self._normalize_density(density_raw)
                loss = self._weighted_negative_log_likelihood(density, accident_counts)
                
                val_loss += loss.item() * distances.size(0)
        
        val_loss /= len(self._valid_dataset)
        self._validLossList.append(val_loss)
        
        # Save best model
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            self._noUpdateEpoch = 0
            if not os.path.exists(self._modelSavePath):
                os.makedirs(self._modelSavePath)
            # Save both networks
            state_dict = {}
            if isinstance(self._stpnn, nn.DataParallel):
                state_dict.update({f'_stpnn.{k}': v for k, v in self._stpnn.module.state_dict().items()})
            else:
                state_dict.update({f'_stpnn.{k}': v for k, v in self._stpnn.state_dict().items()})
            if isinstance(self._stwnn, nn.DataParallel):
                state_dict.update({f'_stwnn.{k}': v for k, v in self._stwnn.module.state_dict().items()})
            else:
                state_dict.update({f'_stwnn.{k}': v for k, v in self._stwnn.state_dict().items()})
            torch.save(state_dict, os.path.join(self._modelSavePath, self._modelName + ".pkl"))
        else:
            self._noUpdateEpoch += 1
    
    def run(self, max_epoch=100, early_stop=-1, **kwargs):
        """Train the model."""
        self.__istrained = True
        
        if self._use_gpu:
            self._stpnn = nn.DataParallel(self._stpnn)
            self._stwnn = nn.DataParallel(self._stwnn)
            self._stpnn = self._stpnn.cuda()
            self._stwnn = self._stwnn.cuda()
        
        # Setup logging
        if not os.path.exists(self._log_path):
            os.makedirs(self._log_path)
        file_str = os.path.join(self._log_path, self._log_file_name)
        logging.basicConfig(
            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            filename=file_str, level=self._log_level)
        
        with tqdm(range(max_epoch)) as pbar:
            for epoch in pbar:
                self._epoch = epoch
                self.__train()
                self.__valid()
                
                pbar.set_postfix({
                    'Train Loss': f"{self._trainLossList[-1]:.5f}",
                    'Valid Loss': f"{self._validLossList[-1]:.5f}",
                    'Best Loss': f"{self._best_loss:.5f}",
                    'LR': f"{self._optimizer.param_groups[0]['lr']:.6f}"
                })
                
                self._scheduler.step()
                
                # TensorBoard logging
                self._writer.add_scalar('Training/Loss', self._trainLossList[-1], self._epoch)
                self._writer.add_scalar('Validation/Loss', self._validLossList[-1], self._epoch)
                self._writer.add_scalar('Training/Learning Rate', 
                                       self._optimizer.param_groups[0]['lr'], self._epoch)
                
                # File logging
                log_str = (f"Epoch: {epoch + 1}; "
                          f"Train Loss: {self._trainLossList[-1]:.5f}; "
                          f"Valid Loss: {self._validLossList[-1]:.5f}; "
                          f"Best Loss: {self._best_loss:.5f}; "
                          f"LR: {self._optimizer.param_groups[0]['lr']:.6f}")
                logging.info(log_str)
                
                # Early stopping
                if 0 < early_stop < self._noUpdateEpoch:
                    print(f"Training stopped! No improvement for {early_stop} epochs.")
                    break
        
        # Load best model
        self.load_model(os.path.join(self._modelSavePath, self._modelName + ".pkl"))
    
    def predict_weights(self, distances, target_context=None, reference_context=None):
        """
        Predict weights for given distances.
        
        Parameters
        ----------
        distances : torch.Tensor or numpy.ndarray
            Spatio-temporal distances (n_samples, n_reference, 2)
        target_context : torch.Tensor or numpy.ndarray, optional
            Contextual features of target arixels (n_samples, context_dim). 
            Required if the model was trained with contextual inputs.
        reference_context : torch.Tensor or numpy.ndarray, optional
            Contextual features of reference arixels (n_samples, n_reference, context_dim).
            Required if the model was trained with contextual inputs.
        
        Returns
        -------
        weights : torch.Tensor
            Normalized weights (n_samples, n_reference) summing to 1
        """
        self._stpnn.eval()
        self._stwnn.eval()
        
        if isinstance(distances, np.ndarray):
            distances = torch.tensor(distances, dtype=torch.float32)
        distances = distances.to(self._device)
        
        if self._context_dim > 0:
            if target_context is not None:
                if isinstance(target_context, np.ndarray):
                    target_context = torch.tensor(target_context, dtype=torch.float32)
                target_context = target_context.to(self._device)
            if reference_context is not None:
                if isinstance(reference_context, np.ndarray):
                    reference_context = torch.tensor(reference_context, dtype=torch.float32)
                reference_context = reference_context.to(self._device)
        else:
            target_context = None
            reference_context = None
        
        with torch.no_grad():
            weights = self._compute_pair_weights(distances, target_context, reference_context)
            if self._use_softmax:
                weights = nn.functional.softmax(weights, dim=1)
            else:
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
        
        return weights.cpu()
    
    def predict_all_arixels(self, dataset=None):
        """
        Predict weights for all arixels (including those without accidents).
        
        Parameters
        ----------
        dataset : STNKDE_Dataset, optional
            Dataset containing all arixels. If None, uses train_dataset.
        
        Returns
        -------
        result_df : pandas.DataFrame
            DataFrame with predicted weights for all arixels
        """
        if dataset is None:
            result = self._train_dataset.get_all_arixels_data()
        else:
            result = dataset.get_all_arixels_data()
        if len(result) == 4:
            all_arixels, all_distances, all_context, ref_context = result
        elif len(result) == 3:
            all_arixels, all_distances, all_context = result
            ref_context = None
        else:
            all_arixels, all_distances = result
            all_context = None
            ref_context = None
        
        # Convert to tensor and predict
        all_distances_tensor = torch.tensor(all_distances, dtype=torch.float32)
        if self._context_dim > 0:
            if all_context is not None:
                target_context_tensor = torch.tensor(all_context, dtype=torch.float32)
            else:
                target_context_tensor = None
            if ref_context is not None:
                # ref_context is already aligned with distances:
                # - If neighbor pruning: (n_samples, k, context_dim) where k is number of neighbors
                # - If no pruning: (n_reference, context_dim) - need to expand to (n_samples, n_reference, context_dim)
                ref_context_tensor = torch.tensor(ref_context, dtype=torch.float32)
                if ref_context_tensor.ndim == 2:
                    # No neighbor pruning: expand to match batch dimension
                    ref_context_tensor = ref_context_tensor.unsqueeze(0).expand(all_distances.shape[0], -1, -1)
                # else: already (n_samples, k, context_dim) - correct shape
            else:
                ref_context_tensor = None
        else:
            target_context_tensor = None
            ref_context_tensor = None
        weights = self.predict_weights(all_distances_tensor, target_context_tensor, ref_context_tensor)
        
        # Sum weights for each arixel (density estimate)
        density = weights.sum(dim=1).numpy()  # (n_all_arixels,)
        
        # Normalize so sum of all densities = 1 (STNKDE requirement)
        density = density / (density.sum() + 1e-10)
        
        result_df = all_arixels.copy()
        result_df['predicted_density'] = density
        result_df['predicted_weight_sum'] = weights.sum(dim=1).numpy()
        
        self.result_data = result_df
        return result_df
    
    def load_model(self, path, map_location=None):
        """Load model from file."""
        if map_location is None:
            map_location = self._device
        state_dict = torch.load(path, map_location=map_location)
        # Handle DataParallel wrapper
        if 'module.' in list(state_dict.keys())[0]:
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # Split state dict for STPNN and STWNN
        stpnn_dict = {k.replace('_stpnn.', ''): v for k, v in state_dict.items() if '_stpnn.' in k}
        stwnn_dict = {k.replace('_stwnn.', ''): v for k, v in state_dict.items() if '_stwnn.' in k}
        if not stpnn_dict and not stwnn_dict:
            # Try direct loading
            self._stpnn.load_state_dict(state_dict, strict=False)
            self._stwnn.load_state_dict(state_dict, strict=False)
        else:
            if stpnn_dict:
                self._stpnn.load_state_dict(stpnn_dict, strict=False)
            if stwnn_dict:
                self._stwnn.load_state_dict(stwnn_dict, strict=False)
        self.__istrained = True
    
    def save_model(self, path=None):
        """Save model to file."""
        if path is None:
            path = os.path.join(self._modelSavePath, self._modelName + ".pkl")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        # Save both networks
        state_dict = {}
        if isinstance(self._stpnn, nn.DataParallel):
            state_dict.update({f'_stpnn.{k}': v for k, v in self._stpnn.module.state_dict().items()})
        else:
            state_dict.update({f'_stpnn.{k}': v for k, v in self._stpnn.state_dict().items()})
        if isinstance(self._stwnn, nn.DataParallel):
            state_dict.update({f'_stwnn.{k}': v for k, v in self._stwnn.module.state_dict().items()})
        else:
            state_dict.update({f'_stwnn.{k}': v for k, v in self._stwnn.state_dict().items()})
        torch.save(state_dict, path)
                
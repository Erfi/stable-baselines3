import torch
from torch import nn as nn
from stable_baselines3.mppi.mbctrl import MBCTRL
from pytorch_mppi import mppi


TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MPPICTRL(MBCTRL):
    def __init__(
        self,
        state_dim,
        action_dim,
        model_in_dim,
        model_out_dim,
        h_units,
        action_ub,
        action_lb,
        state_preproc,
        state_postproc,
        target_proc,
        train_epoch,
        cost_fn,
        noise_sigma,
        lambda_,
        n_samples,
        horizon,
        bootstrap=10,
        true_dynamics_model=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_in_dim = model_in_dim
        self.model_out_dim = model_out_dim
        self.h_units = h_units
        self.action_ub = action_ub
        self.action_lb = action_lb
        self.state_preproc = state_preproc
        self.state_postproc = state_postproc
        self.target_proc = target_proc
        self.train_epoch = train_epoch
        self.cost_fn = cost_fn
        self.noise_sigma = noise_sigma
        self.lambda_ = lambda_
        self.n_samples = n_samples
        self.horizon = horizon
        self.model = true_dynamics_model if true_dynamics_model else self._create_dynamics_model()
        self.use_true_dynamics_model = True if true_dynamics_model else False
        self.ctrl = self._create_MPPI_controller()

    def train(self, states, next_states, actions):
        """
        Trains the dynamic models
        :param states: torch tensor (K, nx)
        :param next_states: torch tensor (K, nx)
        :param acitons: torch tensor (K, nu)
        """

        #  Construct new training points and add to training set
        X = torch.cat([self.state_preproc(states), actions], dim=-1)
        Y = self.target_proc(states, next_states)

        # TODO: Do we really need this thaw ans freeze operations?
        # thaw network
        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(self.model.parameters())
        # loss_fn = torch.nn.MSELoss(reduction="sum")
        for epoch in range(self.train_epoch):
            # TODO: Add batch size here
            # MSE loss
            Yhat = self.model(X)
            # TODO: change to F.MSE ?
            # loss = loss_fn(Yhat, Y)
            loss = (Y - Yhat).norm(2, dim=1) ** 2
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # TODO: use a logger instead
            # print(f"loss: {loss.mean()}")

        # freeze network
        for param in self.model.parameters():
            param.requires_grad = False

    def reset(self):
        """Resets this controller."""
        self.ctrl.reset()

    def act(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        return self.ctrl.command(state)

    def _create_dynamics_model(self):
        network = torch.nn.Sequential(
            torch.nn.Linear(self.model_in_dim, self.h_units),
            torch.nn.Tanh(),
            torch.nn.Linear(self.h_units, self.h_units),
            torch.nn.Tanh(),
            torch.nn.Linear(self.h_units, self.model_out_dim),
        ).to(device=TORCH_DEVICE)
        return network

    def _predict(self, state, action):
        """
        :param state: torch tensor of shape (nx) or (K x nx) current state, or samples of states
        :param action: torch tensor of shape (nu) or (K x nu) current action, of sample of acitons
        :returns next_state predicted by the dynamic model
        """
        # clamp actions
        ubound = torch.ones(action.shape[1]) * self.action_ub
        lbound = torch.ones(action.shape[1]) * self.action_lb
        u = torch.max(torch.min(action, ubound), lbound)
        if state.dim() == 1 or u.dim() == 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        if self.use_true_dynamics_model:
            next_state = self.model(state, action)
        else:
            xu = torch.cat((self.state_preproc(state), u), dim=1)
            state_residual = self.model(xu)
            next_state = self.state_postproc(state, state_residual)
        return next_state

    def _create_MPPI_controller(self):
        ctrl = mppi.MPPI(
            dynamics=self._predict,
            running_cost=self.cost_fn,
            nx=self.state_dim,
            noise_sigma=self.noise_sigma,
            num_samples=self.n_samples,
            horizon=self.horizon,
            lambda_=self.lambda_,
            device=TORCH_DEVICE,
            u_min=torch.tensor(self.action_lb, dtype=torch.float, device=TORCH_DEVICE),
            u_max=torch.tensor(self.action_ub, dtype=torch.float, device=TORCH_DEVICE),
        )
        return ctrl

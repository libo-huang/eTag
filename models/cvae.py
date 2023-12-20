#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from torch import nn
import torch, sys
sys.path.append("..")

from utils.utils import idx2onehot


class CVAE_Mnist_v0(nn.Module):

    def __init__(self, args):
        super(CVAE_Mnist, self).__init__()
        self.feat_dim = args.feat_dim
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.class_dim = args.class_dim

        self.model_encoder = nn.Sequential(
            nn.Linear(self.feat_dim+self.class_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(), 
        )
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dim, self.latent_dim)
        self.model_decoder = nn.Sequential(
            nn.Linear(self.latent_dim+self.class_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.feat_dim),
        )
        self.sigmoid = nn.LeakyReLU()  # 因为Mnist像素值在0-1之间，所以建议用Sigmoid(), tanh()

    def encode(self, x, y):
        x = torch.cat((x, y), 1)
        x = self.model_encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar

    def decode(self, z, y):
        z = torch.cat((z, y), 1)
        logit = self.model_decoder(z)
        feat = self.sigmoid(logit)
        return feat

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(std.device)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x, y):
        x = x.reshape(x.shape[0], -1)
        y = torch.eye(self.class_dim)[y].to(x.device)
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z, y).reshape(x.shape[0], 1, 32, 32)

        return x_rec, mu, logvar
        # test
        # x = torch.randn(2, 1, 32, 32)
        # y = torch.randint(0, 9, [2, ])
        #
        # class arg(object):
        #     def __init__(self):
        #         self.feat_dim = 32 * 32
        #         self.latent_dim = 2
        #         self.hidden_dim = 100
        #         self.class_dim = 10
        #
        # args = arg()
        # model = CVAE_Mnist(args)
        # x_rec, _, _ = model(x, y)


class CVAE_Cifar(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.encoder_layer_sizes = args.encoder_layer_sizes
        self.latent_size = args.latent_size
        self.decoder_layer_sizes = args.decoder_layer_sizes
        self.num_labels = args.class_dim

        self.encoder = Encoder(self.encoder_layer_sizes, self.latent_size, self.num_labels)
        self.decoder = Decoder(self.decoder_layer_sizes, self.latent_size, self.num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 32*32)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, num_labels):

        super().__init__()

        # layer_sizes[0] += num_labels
        self.num_labels = num_labels
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip([layer_sizes[0] + num_labels] + layer_sizes[1:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        c = idx2onehot(c, n=self.num_labels)
        x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, num_labels):

        super().__init__()
        self.num_labels = num_labels
        self.MLP = nn.Sequential()

        input_size = latent_size + num_labels

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                # self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
                self.MLP.add_module(name="sigmoid", module=nn.ReLU())

    def forward(self, z, c):

        c = idx2onehot(c, n=self.num_labels)
        z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)
        # x = x.view(x.size(0), 1, 32, 32)
        return x


if __name__ == '__main__':
    x = torch.randn(2, 512)
    y = torch.randint(0, 9, [2, ])

    class arg(object):
        def __init__(self):
            self.encoder_layer_sizes = [512, 512, 512]
            self.decoder_layer_sizes = [512, 512, 512]
            self.latent_size = 200
            self.class_dim = 100
    args = arg()
    model = CVAE_Cifar(args)
    x_rec, mean, log_var, z = model(x, y)
    print('end')
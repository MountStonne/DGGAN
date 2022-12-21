import argparse
from normalization import *
from onehotencoding import *
from pathlib import Path
from modules import *
import torchvision.transforms as trans
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import warnings
warnings.filterwarnings('ignore')


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def run(
        data_path=ROOT / 'data' / 'olympics.csv',
        continuous_columns=['Age','Height', 'Weight'],
        categorical_columns=['Sex', 'Year', 'Season', 'City', 'Sport', 'Medal', 'AOS', 'AOE'],
        device="cpu",
        batch_size=32,
        epochs=50,
        lr=0.0001,
        ns_G=0.8,
        ns_D=0.1,
        amount=1,
):
    # Load data
    df = pd.read_csv(data_path)

    # Find continuous data and normalization
    df_conti = df[continuous_columns].astype('int64')
    norm_list = continuous_columns
    norm_types = ['standard' for i in range(len(norm_list))]
    df_conti_norm, dict_conti = norm(df_conti, norm_list, norm_types)

    # Find categorical data and one hot encoding
    df_category = df[categorical_columns].astype('category')
    cate_name, cate_class_number, cate_class, df_category_ohe = one_hot_encoding(df_category)

    # Combine data
    df_combine = pd.concat([df_conti_norm, df_category_ohe], axis=1)

    # Reshape data
    df_length = len(df_combine.columns)
    input_data = df_combine.to_numpy().flatten().reshape(-1, 1, 1, df_length)

    # ToTensor transform
    transforms = trans.Compose(
        [trans.ToTensor()]
    )

    # Parameters
    z_dim = 64
    image_dim = 1 * df_length * 1
    num_epochs = epochs
    if num_epochs == 200:
        a = 1e-2 / 100
        r = 1.02872
    elif num_epochs == 100:
        a = 1e-2 / 100
        r = 1.0673
    elif num_epochs == 50:
        a = 1e-2 / 100
        r = 1.15884

    # Model initialization
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    dataset = OlympicDataset(input_data, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    disc = Discriminator(image_dim, ns_D).to(device)
    gen = Generator(z_dim, image_dim, ns_G).to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    df_real_continuous = df[df_conti.columns.to_numpy()].astype('int64')
    df_real_categorical = df[df_category.columns.to_numpy()].astype('category')
    df_real = pd.concat([df_real_continuous, df_real_categorical], axis=1)

    # Train and generate
    step = 0
    print('Process:')
    for epoch in range(num_epochs):
        for batch_idx, real in enumerate(loader):

            real = real.view(-1, 1 * df_length).to(device)
            batch_size = real.shape[0]

            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # where the second option of maximizing doesn't suffer from
            # saturating gradients
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 1, df_length)
                    step += 1

        if epoch == 0:
            fake_df = pd.DataFrame(fake.flatten().reshape(-1, df_length).detach().numpy())
            fake_df = fake_df.rename(columns={i: df_combine.columns[i] for i in range(df_combine.columns.shape[0])})
            final = fake_df.astype('category')
            loop_num = int(len(df_real) * a // len(fake_df))
            for i in range(1 if loop_num == 0 else loop_num):
                noise = torch.randn(batch_size, z_dim).to(device)
                fake = gen(noise)
                fake_df = pd.DataFrame(fake.flatten().reshape(-1, df_length).detach().numpy())
                fake_df = fake_df.rename(columns={i: df_combine.columns[i] for i in range(df_combine.columns.shape[0])})
                demo = fake_df
                final = pd.concat([final, demo]).reset_index(drop=True)
        else:
            loop_num = int(len(df_real) * a // len(fake_df) * amount)
            for i in range(loop_num + 1):
                noise = torch.randn(batch_size, z_dim).to(device)
                fake = gen(noise)
                fake_df = pd.DataFrame(fake.flatten().reshape(-1, df_length).detach().numpy())
                fake_df = fake_df.rename(columns={i: df_combine.columns[i] for i in range(df_combine.columns.shape[0])})
                demo = fake_df
                final = pd.concat([final, demo]).reset_index(drop=True)

        a = a * r

    # Save generated results
    df_fake_categorical = one_hot_decoding(final.iloc[:, len(continuous_columns):])[categorical_columns].astype(
        'category')
    df_fake_continuous = denorm(final[continuous_columns], norm_list, norm_types, dict_conti).apply(np.ceil).astype(
        'int64')
    df_fake = pd.concat([df_fake_continuous, df_fake_categorical], axis=1)
    df_fake.to_csv(ROOT / 'generations/generation.csv', index=False)
    print('Synthetic data has been saved to ', ROOT / 'generations/generation.csv')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=ROOT / 'data' / 'olympics.csv', help='source of data file')
    parser.add_argument('--continuous_columns', nargs="*", type=str, default=['Age','Height', 'Weight'],
                        help='list of continuous columns')
    parser.add_argument('--categorical_columns', nargs="*", type=str,
                        default=['Sex', 'Year', 'Season', 'City', 'Sport', 'Medal', 'AOS', 'AOE'],
                        help='list of categorical columns')
    parser.add_argument('--device', type=str, default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--ns_G', type=float, default=0.8, help='leakyRelu negative slope of generator')
    parser.add_argument('--ns_D', type=float, default=0.1, help='leakyRelu negative slope of discriminator')
    parser.add_argument('--amount', type=float, default=1, help='percentage of generated data size over real data size')

    opt = parser.parse_args()

    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

# 导入相关包
import torchvision
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from model import *
from mnist_data import *
from model import Generator, Discriminator
from dataset import image_data_set
class anogan_trainer:
    '''
    trainer class for AnoGAN
    '''
    def __init__(self,args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # batch_size默认128
        self.batch_size = args.batch_size
        self.train_set = image_data_set(train)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        # 后续解包更多参数【】

        self.G = Generator().to(self.device)
        self.D = Discriminator().to(self.device)

    def train(self):
        # 加载模型
        
        # 训练模式
        self.G.train()
        self.D.train()

        # 设置优化器
        optimizerG = torch.optim.Adam(self.G.parameters(), lr=0.0001, betas=(0.0, 0.9))
        optimizerD = torch.optim.Adam(self.D.parameters(), lr=0.0004, betas=(0.0, 0.9))

        # 定义损失函数
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        """
        训练
        """
        # 开始训练
        for epoch in range(self.args.epochs):
            # 定义初始损失
            log_g_loss, log_d_loss = 0.0, 0.0
            for images in self.train_loader:
                images = images.to(self.device)

                ## 训练判别器 Discriminator
                # 定义真标签（全1）和假标签（全0）   维度：（batch_size）
                label_real = torch.full((images.size(0),), 1.0).to(self.device)
                label_fake = torch.full((images.size(0),), 0.0).to(self.device)

                # 定义潜在变量z    维度：(batch_size,20,1,1)
                z = torch.randn(images.size(0), 20).to(self.device).view(images.size(0), 20, 1, 1).to(self.device)
                # 潜在变量喂入生成网络--->fake_images:(batch_size,1,64,64)
                fake_images = self.G(z)

                # 真图像和假图像送入判别网络，得到d_out_real、d_out_fake   维度：都为（batch_size,1,1,1）
                d_out_real, _ = self.D(images)
                d_out_fake, _ = self.D(fake_images)

                # 损失计算
                d_loss_real = criterion(d_out_real.view(-1), label_real)
                d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
                d_loss = d_loss_real + d_loss_fake

                # 误差反向传播，更新损失
                optimizerD.zero_grad()
                d_loss.backward()
                optimizerD.step()

                ## 训练生成器 Generator
                # 定义潜在变量z    维度：(batch_size,20,1,1)
                z = torch.randn(images.size(0), 20).to(self.device).view(images.size(0), 20, 1, 1).to(self.device)
                fake_images = self.G(z)

                # 假图像喂入判别器，得到d_out_fake   维度：（batch_size,1,1,1）
                d_out_fake, _ = self.D(fake_images)

                # 损失计算
                g_loss = criterion(d_out_fake.view(-1), label_real)

                # 误差反向传播，更新损失
                optimizerG.zero_grad()
                g_loss.backward()
                optimizerG.step()

                ## 累计一个epoch的损失，判别器损失和生成器损失分别存放到log_d_loss、log_g_loss中
                log_d_loss += d_loss.item()
                log_g_loss += g_loss.item()

            ## 打印损失
            print(f'epoch {epoch}, D_Loss:{log_d_loss / 128:.4f}, G_Loss:{log_g_loss / 128:.4f}')




        ## 展示生成器存储的图片，存放在result文件夹下的G_out.jpg
        z = torch.randn(8, 20).to(self.device).view(8, 20, 1, 1).to(self.device)
        fake_images = self.G(z)
        torchvision.utils.save_image(fake_images,f"result/G_out.jpg")

    
    def anomaly_score(self,input_image, fake_image, D):
        '''
        定义缺陷计算的得分
        '''
        # Residual loss 计算
        residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

        # Discrimination loss 计算
        _, real_feature = D(input_image)
        _, fake_feature = D(fake_image)
        discrimination_loss = torch.sum(torch.abs(real_feature - fake_feature), (1))

        # 结合Residual loss和Discrimination loss计算每张图像的损失
        total_loss_by_image = 0.9 * residual_loss + 0.1 * discrimination_loss
        # 计算总损失，即将一个batch的损失相加
        total_loss = total_loss_by_image.sum()

        return total_loss, total_loss_by_image, residual_loss
    
    def test(self):
        """
        测试
        """
        # 加载测试数据
        test_set = image_data_set(test)
        test_loader = DataLoader(test_set, batch_size=5, shuffle=False)
        input_images = next(iter(test_loader)).to(self.device)

        # 定义潜在变量z  维度：（5，20，1，1）
        z = torch.randn(5, 20).to(self.device).view(5, 20, 1, 1)
        # z的requires_grad参数设置成Ture,让z可以更新
        z.requires_grad = True
        # 定义优化器
        z_optimizer = torch.optim.Adam([z], lr=1e-3)
        # 搜索z
        for epoch in range(5000):
            fake_images = self.G(z)
            loss, _, _ = self.anomaly_score(input_images, fake_images, self.D)

            z_optimizer.zero_grad()
            loss.backward()
            z_optimizer.step()

            if epoch % 1000 == 0:
                print(f'epoch: {epoch}, loss: {loss:.0f}')




        fake_images = self.G(z)

        _, total_loss_by_image, _ = self.anomaly_score(input_images, fake_images, self.D)

        print(total_loss_by_image.cpu().detach().numpy())

        torchvision.utils.save_image(input_images, f"result/Nomal.jpg")
        torchvision.utils.save_image(fake_images, f"result/ANomal.jpg")


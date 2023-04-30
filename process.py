from torch.utils.data import DataLoader

from unet import *
from mydataset import *


class Processor:
    def __init__(self, train_path, eval_path, batch_size=1, epochs=50):
        self.train_path = train_path
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.epochs = epochs

        # GPU/CPU
        self.device = torch.device("cpu")

        # 数据
        self.train_data = MyDataset(os.path.join(self.train_path, 'images'), os.path.join(self.train_path, 'labels'))
        self.eval_data = MyDataset(os.path.join(self.eval_path, 'images'), os.path.join(self.eval_path, 'labels'))
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.eval_loader = DataLoader(self.eval_data, batch_size=self.batch_size, shuffle=True)
        print("train data size:{}, eval data size:{}".format(len(self.train_data), len(self.eval_data)))

        # 模型、优化器、损失函数
        self.model = UNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.BCELoss()

    def train(self, epoch):
        self.model.train()
        epoch_loss = 0
        num = 0
        print("start epoch:{}".format(epoch))

        for img, label in self.train_loader:
            img = img.to(self.device)
            label = label.to(self.device)
            # 梯度清零
            self.optimizer.zero_grad()
            # 前向传播
            output = self.model(img)
            # 计算损失
            loss = self.loss_fn(output, label)
            print("image:{}, loss:{}".format(num, loss.item()))
            epoch_loss += loss.item()
            # 反向传播
            loss.backward()
            # 更新参数
            self.optimizer.step()
            num += 1

        print("epoch:{}, avg loss:{}".format(epoch, epoch_loss / num))
        torch.save(self.model.state_dict(), "./model/{}.pth".format(epoch))

    def eval(self):
        self.model.eval()
        total_loss = 0
        num = 0

        for img, label in self.eval_loader:
            img = img.to(self.device)
            label = label.to(self.device)
            # 前向传播
            output = self.model(img)
            # 计算损失
            loss = self.loss_fn(output, label)
            print("image:{}, loss:{}".format(num, loss.item()))
            total_loss += loss.item()
            num += 1

        print("avg loss:{}".format(total_loss / num))
        return total_loss / num

    def extract(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            if epoch % 5 == 0:
                loss = self.eval()
                if loss < 0.01:
                    break

    def process(self, img_path):
        img = Image.open(img_path)
        img = img.convert("L").resize((1024, 768))
        img = TF.to_tensor(img).unsqueeze(0)
        img = img.to(self.device)
        self.model.eval()
        output = self.model(img).squeeze(0)
        return TF.to_pil_image(output)

from process import Processor


def main():
    m = Processor('./dataset/train', './dataset/eval')
    m.extract('./model/20.pth')
    img = m.process('./dataset/eval/images/IM_0024.jpg')
    img.save('./outputs/IM_0024.png')


if __name__ == '__main__':
    main()

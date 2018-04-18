clear;clc;
%提前申请空间，加快计算速度
HOGtrain = zeros(30461,2916);
HOGtest  = zeros(9856,2916);
TabTrain = zeros(30461,1);
TabTest  = zeros(9856,1);

%训练样本集为图片
sTrainNameBasePath = 'D:\Database\Training_Images_Scale';
sTestNameBasePath = 'D:\Database\Test_Images_Scale';

j = 0;
%读入训练样本：共43类，从0到42，首先转为灰度图再提取HOG特征
for nTrainNumFolder = 0:42
    sTrainFolder = num2str(nTrainNumFolder, '%05d');
    sTrainNamePath = [sTrainNameBasePath, '\', sTrainFolder];
    sTrainAnyJPGpath = [sTrainNamePath,'\*.jpg'];
    listTrain = dir(sTrainAnyJPGpath);
    m = size(listTrain,1);
    %遍历每一个文件夹下的所有图片
    for TrainOneClassNum = 0:m-1
        sTrainJPG = num2str(TrainOneClassNum, '%05d');
        sTrainJPGpath = [sTrainNamePath,'\',sTrainJPG,'.jpg'];
        
        TrainImage = rgb2gray(imread(sTrainJPGpath));
%		  加上直方图均衡化，SVM训练通过率有所下降
%         直方图均衡化
%         b = mean(TrainImage(:));
%         if b<45
%             TrainImageHistEq = histeq(TrainImage,256);
%         else
%             TrainImageHistEq = TrainImage;
%         end
%         
%         %[TrainImageWiener,noise] = wiener2(TrainImageHistEq);
        
        v = hogcalculator(TrainImageHistEq);    %行向量
        j = j + 1;
        HOGtrain(j, :) = v;
        TabTrain(j) = nTrainNumFolder;
    end
end

%%PCA降维
[COEFF,SCORE,latent,tsquare] = princomp(HOGtrain);
b = zeros(2916,1);
for i = 1:numel(latent)
   b = cumsum(latent)./sum(latent);
end

for k = 1:numel(b)
    if (b(k)>0.95)
        c = k;
        break;
    end
end

COEFF_PCA = zeros(2916, c);
for l = 1:c
    COEFF_PCA(:,l) = COEFF(:,l);
end

train = HOGtrain * COEFF_PCA;
train = train';
[train_scale,ps] = mapminmax(train,0,1);
train_scale = train_scale';

%%读入测试样本
j = 0;
%读入训练样本
for nTestNumFolder = 0:42
    sTestFolder = num2str(nTestNumFolder, '%05d');
    sTestNamePath = [sTestNameBasePath, '\', sTestFolder];
    sTestAnyJPGpath = [sTestNamePath,'\*.jpg'];
    listTest = dir(sTestAnyJPGpath);
    n = size(listTest,1);
    %子类内历遍
    for TestOneClassNum = 0:n-1
        sTestJPG = num2str(TestOneClassNum, '%05d');
        sTestJPGpath = [sTestNamePath,'\',sTestJPG,'.jpg'];
        
        TestImage = rgb2gray(imread(sTestJPGpath));
%         直方图均衡化        
%         c = mean(TestImage(:));
%         if c<45
%             TestImageHistEq = histeq(TestImage,256);
%         else
%             TestImageHistEq = TestImage;
%         end
%         
%         %[TestImageWiener,noise] = wiener2(TestImageHistEq);
        
        w = hogcalculator(TestImageHistEq);    %行向量
        j = j + 1;
        HOGtest(j, :) = w;
        TabTest(j) = nTestNumFolder;
    end
end
j = 0;

%测试集需要同样的PCA降维
test_PCA = HOGtest * COEFF_PCA;
test_PCA = test_PCA';
[test_scale,ps] = mapminmax(test_PCA,0,1);
test_scale = test_scale';

%% SVM网络训练
model = svmtrain(TabTrain, train_scale, '-c 100 -g 32 -t 0');

%% SVM分类
[predict_label, accuracy,t] = svmpredict(TabTest, test_scale, model);
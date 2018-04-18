clear;clc;
%��ǰ����ռ䣬�ӿ�����ٶ�
HOGtrain = zeros(30461,2916);
HOGtest  = zeros(9856,2916);
TabTrain = zeros(30461,1);
TabTest  = zeros(9856,1);

%ѵ��������ΪͼƬ
sTrainNameBasePath = 'D:\Database\Training_Images_Scale';
sTestNameBasePath = 'D:\Database\Test_Images_Scale';

j = 0;
%����ѵ����������43�࣬��0��42������תΪ�Ҷ�ͼ����ȡHOG����
for nTrainNumFolder = 0:42
    sTrainFolder = num2str(nTrainNumFolder, '%05d');
    sTrainNamePath = [sTrainNameBasePath, '\', sTrainFolder];
    sTrainAnyJPGpath = [sTrainNamePath,'\*.jpg'];
    listTrain = dir(sTrainAnyJPGpath);
    m = size(listTrain,1);
    %����ÿһ���ļ����µ�����ͼƬ
    for TrainOneClassNum = 0:m-1
        sTrainJPG = num2str(TrainOneClassNum, '%05d');
        sTrainJPGpath = [sTrainNamePath,'\',sTrainJPG,'.jpg'];
        
        TrainImage = rgb2gray(imread(sTrainJPGpath));
%		  ����ֱ��ͼ���⻯��SVMѵ��ͨ���������½�
%         ֱ��ͼ���⻯
%         b = mean(TrainImage(:));
%         if b<45
%             TrainImageHistEq = histeq(TrainImage,256);
%         else
%             TrainImageHistEq = TrainImage;
%         end
%         
%         %[TrainImageWiener,noise] = wiener2(TrainImageHistEq);
        
        v = hogcalculator(TrainImageHistEq);    %������
        j = j + 1;
        HOGtrain(j, :) = v;
        TabTrain(j) = nTrainNumFolder;
    end
end

%%PCA��ά
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

%%�����������
j = 0;
%����ѵ������
for nTestNumFolder = 0:42
    sTestFolder = num2str(nTestNumFolder, '%05d');
    sTestNamePath = [sTestNameBasePath, '\', sTestFolder];
    sTestAnyJPGpath = [sTestNamePath,'\*.jpg'];
    listTest = dir(sTestAnyJPGpath);
    n = size(listTest,1);
    %����������
    for TestOneClassNum = 0:n-1
        sTestJPG = num2str(TestOneClassNum, '%05d');
        sTestJPGpath = [sTestNamePath,'\',sTestJPG,'.jpg'];
        
        TestImage = rgb2gray(imread(sTestJPGpath));
%         ֱ��ͼ���⻯        
%         c = mean(TestImage(:));
%         if c<45
%             TestImageHistEq = histeq(TestImage,256);
%         else
%             TestImageHistEq = TestImage;
%         end
%         
%         %[TestImageWiener,noise] = wiener2(TestImageHistEq);
        
        w = hogcalculator(TestImageHistEq);    %������
        j = j + 1;
        HOGtest(j, :) = w;
        TabTest(j) = nTestNumFolder;
    end
end
j = 0;

%���Լ���Ҫͬ����PCA��ά
test_PCA = HOGtest * COEFF_PCA;
test_PCA = test_PCA';
[test_scale,ps] = mapminmax(test_PCA,0,1);
test_scale = test_scale';

%% SVM����ѵ��
model = svmtrain(TabTrain, train_scale, '-c 100 -g 32 -t 0');

%% SVM����
[predict_label, accuracy,t] = svmpredict(TabTest, test_scale, model);
clc
clear
close all
%%
load('encode_data_bank.mat')

% [num, txt, raw] = xlsread('bank-additional-full.xlsx');
%
% Y = zeros(length(raw) - 1,1);
% for i = 2:length(raw)
%     if strcmp(raw(i,16),'no') == 1
%         Y(i - 1,1) = 0;
%     else
%         Y(i - 1,1) = 1;
%     end
% end
%
% X = zeros(length(raw) - 1,15);

% 1)age	2)job 3)marital	4)education	5)default 6)housing	7)loan 8)contact 9)month
% 10)day_of_week 11)duration 11)campaign 12)pdays 13)previous 14)poutcome 15)y
X = double(data);

X(:,1) = mod(X(:,1),10); % zero outlier 1%

X(:,11) = []; % filloutliers(X(:,11),'clip','quartiles'); % 7% to zeo

X(:,12) = (X(:,12) - mean(X(:,12))) ./ (max(X(:,12)) - min(X(:,12)));

figure(1)
plot(X(Y == 1,1),'*b')
hold on
plot(X(Y == 0,1),'or')
hold on

figure(2)
labels = {'Yes','No'};
pie([sum(Y)/length(Y), 1 - sum(Y)/length(Y)], labels)








% random = '';
%
% temp = zeros(11,7);
%
% % 1) x-axis 2) F1 3) Accuracy
% load(strcat('result_LR',random,'.mat'))
%
% figure(1)
% plot(result(:,1),100 * result(:,2),'-+b')
% hold on
%
% figure(2)
% plot(result(:,1),100 * result(:,3),'-+b')
% hold on
%
% temp(:,1) = result(:,1);
% temp(:,2) = result(:,2);
%
% load(strcat('result_LS',random,'.mat'))
%
% figure(1)
% plot(result(:,1),100 * result(:,2),'-*y')
% hold on
%
% figure(2)
% plot(result(:,1),100 * result(:,3),'-*y')
% hold on
%
% temp(:,3) = result(:,2);
%
% load(strcat('result_KNN',random,'.mat'))
%
% figure(1)
% plot(result(:,1),100 * result(:,2),'-og')
% hold on
%
% figure(2)
% plot(result(:,1),100 * result(:,3),'-og')
% hold on
%
% temp(:,4) = result(:,2);
%
% load(strcat('result_DT',random,'.mat'))
%
% figure(1)
% plot(result(:,1),100 * result(:,2),'-sc')
% hold on
%
% figure(2)
% plot(result(:,1),100 * result(:,3),'-sc')
% hold on
%
% temp(:,5) = result(:,2);
%
% load(strcat('result_RF',random,'.mat'))
%
% figure(1)
% plot(result(:,1),100 * result(:,2),'-xr')
% hold on
%
% figure(2)
% plot(result(:,1),100 * result(:,3),'-xr')
% hold on
%
% temp(:,6) = result(:,2);
%
% load(strcat('result_GNB',random,'.mat'))
%
% figure(1)
% plot(result(:,1),100 * result(:,2),'-*m')
% hold on
% legend('Logistic Regression','Linear SVM','KNN','Decision Tree','Random Forest','Gaussian NB')
% xlabel('Percentage of Train Data')
% ylabel('F1 Accuracy (weighted)')
% ylim([65 90])
%
% figure(2)
% plot(result(:,1),100 * result(:,3),'-*m')
% hold on
% legend('Logistic Regression','Linear SVM','KNN','Decision Tree','Random Forest','Gaussian NB')
% xlabel('Percentage of Train Data')
% ylabel('Accuracy')
% ylim([65 90])
%
% temp(:,7) = result(:,2);
%
% xlswrite('bank.xlsx',temp)
clc
clear
close all

random = '';

temp = zeros(11,7);

% 1) x-axis 2) F1 3) Accuracy
load(strcat('result_LR',random,'.mat'))

figure(1)
plot(result(:,1),100 * result(:,2),'-+b')
hold on

figure(2)
plot(result(:,1),100 * result(:,3),'-+b')
hold on

temp(:,1) = result(:,1);
temp(:,2) = result(:,2);

load(strcat('result_LS',random,'.mat'))

figure(1)
plot(result(:,1),100 * result(:,2),'-*y')
hold on

figure(2)
plot(result(:,1),100 * result(:,3),'-*y')
hold on

temp(:,3) = result(:,2);

load(strcat('result_KNN',random,'.mat'))

figure(1)
plot(result(:,1),100 * result(:,2),'-og')
hold on

figure(2)
plot(result(:,1),100 * result(:,3),'-og')
hold on

temp(:,4) = result(:,2);

load(strcat('result_DT',random,'.mat'))

figure(1)
plot(result(:,1),100 * result(:,2),'-sc')
hold on

figure(2)
plot(result(:,1),100 * result(:,3),'-sc')
hold on

temp(:,5) = result(:,2);

load(strcat('result_RF',random,'.mat'))

figure(1)
plot(result(:,1),100 * result(:,2),'-xr')
hold on

figure(2)
plot(result(:,1),100 * result(:,3),'-xr')
hold on

temp(:,6) = result(:,2);

load(strcat('result_GNB',random,'.mat'))

figure(1)
plot(result(:,1),100 * result(:,2),'-*m')
hold on
legend('Logistic Regression','Linear SVM','KNN','Decision Tree','Random Forest','Gaussian NB')
xlabel('Percentage of Train Data')
ylabel('F1 Accuracy (weighted)')
ylim([65 90])

figure(2)
plot(result(:,1),100 * result(:,3),'-*m')
hold on
legend('Logistic Regression','Linear SVM','KNN','Decision Tree','Random Forest','Gaussian NB')
xlabel('Percentage of Train Data')
ylabel('Accuracy')
ylim([65 90])

temp(:,7) = result(:,2);

xlswrite('bank.xlsx',temp)
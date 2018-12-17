clc
clear
close all

data  = xlsread('ResultsForBikeAssign2.xlsx');

figure(1)
plot(100 * data(:,1),data(:,3), '-+b', 'LineWidth', 1.2)
hold on
plot(100 * data(:,1),data(:,4), '-xr', 'LineWidth', 1.2)
hold on
plot(100 * data(:,1),data(:,5),'-sg', 'LineWidth', 1.2)
hold on
plot(100 * data(:,1),data(:,6),'-oc', 'LineWidth', 1.2)
hold on
plot(100 * data(:,1),data(:,7),'-*m', 'LineWidth', 1.2)
hold on

% yVector	yVectorRF	yVectorSvm	yVectorRid	yVectorEns
legend('Linear Regression','Random Forest','SV Regression','Ridge Regression','Ensemble')
xlabel('Percentage of Train Data')
ylabel('MAPE')
ylim([0 350])
grid on
clc
clear
close all
%%
load('encode_data_bank.mat')

X = double(data);
Y = X(:,16); X(:,16) = [];

X(:,11) = []; % filloutliers(X(:,11),'clip','quartiles'); % 7% to zeo

% 1)age	2)job 3)marital	4)education	5)default 6)housing	7)loan 8)contact 9)month
% 10)day_of_week >> 11)duration 11)campaign 12)pdays 13)previous 14)poutcome

X(:,1) = floor(X(:,1) ./ 10) * 10;

% X(:,12) = (X(:,12) - mean(X(:,12))) ./ (max(X(:,12)) - min(X(:,12)));

t1 = unique(X(:,1)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,1) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,1) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,1) == t1(i))) / length(X);
    k = 0;
end

figure(1)
bar(t1,t2(:,1))
hold on
xlabel('Age')
ylabel('Percentage of Clients Subscribed a Term Deposit (%)')

figure(2)
bar(t1,t2(:,2), 'r');
hold on
xlabel('Age')
ylabel('Percentage of Dataset')

t1 = unique(X(:,2)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,2) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,2) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,2) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'Admin','Blue-collar','Entrepreneur','Housemaid','Management','Retired',...
    'Self-employed','Services','Student','Technician','Unemployed','Unknown'});

figure(3)
bar(c,t2(:,1))
hold on
xlabel('Type of Job')
ylabel('Percentage of Clients Subscribed a Term Deposit (%)')

figure(4)
bar(c,t2(:,2), 'r');
hold on
xlabel('Type of Job')
ylabel('Percentage of Dataset')

t1 = unique(X(:,3)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,3) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,3) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,3) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'divorced','married','single','unknown'});

figure(5)
bar(c,t2)
hold on
xlabel('Marital Status')
ylabel('(%)')
legend('Percentage of Clients Subscribed a Term Deposit','Percentage of Total Dataset')

t1 = unique(X(:,5)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,5) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,5) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,5) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'no','unknown','yes'});

figure(6)
bar(c,t2)
hold on
xlabel('Credit in Default')
ylabel('(%)')
legend('Percentage of Clients Subscribed a Term Deposit','Percentage of Total Dataset')

t1 = unique(X(:,6)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,6) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,6) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,6) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'no','unknown','yes'});

figure(7)
bar(c,t2)
hold on
xlabel('Housing Loan')
ylabel('(%)')
legend('Percentage of Clients Subscribed a Term Deposit','Percentage of Total Dataset')


t1 = unique(X(:,7)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,7) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,7) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,7) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'no','unknown','yes'});

figure(8)
bar(c,t2)
hold on
xlabel('Personal Loan')
ylabel('(%)')
legend('Percentage of Clients Subscribed a Term Deposit','Percentage of Total Dataset')

t1 = unique(X(:,8)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,8) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,8) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,8) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'Cellular','Telephone'});

figure(9)
bar(c,t2)
hold on
xlabel('Contact Communication Type')
ylabel('(%)')
legend('Percentage of Clients Subscribed a Term Deposit','Percentage of Total Dataset')

t1 = unique(X(:,9)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,9) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,9) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,9) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'Apr','Aug','Dec','Jun','Jul','Mar','May','Nov','Oct','Sep'});

figure(10)
bar(c,t2(:,1))
hold on
xlabel('Last Contact Month of Year')
ylabel('Percentage of Clients Subscribed a Term Deposit (%)')

figure(11)
bar(c,t2(:,2), 'r');
hold on
xlabel('Last Contact Month of Year')
ylabel('Percentage of Dataset')

% figure(10)
% bar(c,t2,'group')
% hold on
% xlabel('Last Contact Month of Year')
% ylabel('(%)')
% legend('Percentage of Clients Subscribed a Term Deposit','Percentage of Total Dataset')

t1 = unique(X(:,10)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,10) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,10) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,10) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'Friday','Monday','Thursday','Tuesday','Wednesday'});

figure(12)
bar(c,t2)
hold on
xlabel('Last Contact Day of The Week')
ylabel('(%)')
legend('Percentage of Clients Subscribed a Term Deposit','Percentage of Total Dataset')

t1 = unique(X(:,4)); t2 = zeros(1,length(t1));
k = 0;
for i = 1:length(t1)
    for j = 1:length(X)
        if (X(j,4) == t1(i))
            if (Y(j) == 1)
                k = k + 1;
            end
        end
    end
    t2(i,1) = 100 * k / length(find(X(:,4) == t1(i)));
    t2(i,2) = 100 * length(find(X(:,4) == t1(i))) / length(X);
    k = 0;
end

c = categorical({'Basic (4y)','Basic (6y)','Basic (9y)','High School','Illiterate',...
    'Professional Course','University Degree','Unknown'});

figure(13)
bar(c,t2(:,1))
hold on
xlabel('Education')
ylabel('Percentage of Clients Subscribed a Term Deposit (%)')

figure(14)
bar(c,t2(:,2), 'r');
hold on
xlabel('Education')
ylabel('Percentage of Dataset')
%%
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
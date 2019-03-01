clear;clc;
eval_=zeros(15,2);

max_min = csvread('/home/cyan/data/human3.6m/annotation/16point_mean_limb_scaled_max_min.csv');

% s = '%s%f';
% for i=1:80
%    s=strcat(s,'%f');
% end

s = '%s%f';
for i=1:48
   s=strcat(s,'%f');
end

for folder_i = 1:15
    action_id = num2str(folder_i);

% % action_id='9';
root_folder1='/data/h36m/p3-mask';
root_folder2='/data/h36m/test_p3/';
fliename = fopen(strcat(root_folder2,'test',action_id,'.txt'));

fliename2 = strcat('result',action_id,'_unnorm.csv');
fliename3 = strcat('result',action_id,'_change_unnorm.csv');

pred_=csvread(fullfile(root_folder1,fliename2));
change_=csvread(fullfile(root_folder1,fliename3));

gt2d3d = textscan(fliename,s,'delimiter',[' ',',']);
filename = gt2d3d{1,1};
gt2d3d{1,1}=[];
gt2d3d = cell2mat(gt2d3d);

gt = gt2d3d(:,1:end-1);
gt = gt .* repmat(max_min(1,:)-max_min(2,:),[size(gt,1),1]) + repmat(max_min(2,:),[size(gt,1),1]);
[num_samp,num_ord]=size(pred_);
pred= pred_(:,2:end);
change= change_(:,2:end);
num_ord=num_ord-1;
num_skel=num_ord/3;

sqrt_error = zeros(num_samp,1);
sqrt_error2 = zeros(num_samp,1);
count = 1;
protocp2 = {};
for i=1:num_samp
        protocp2{count} = filename{i,1};
        for j=0:num_skel-1
            x=pred(i,3*j+1)-gt(i,3*j+1);
            y=pred(i,3*j+2)-gt(i,3*j+2);
            z=pred(i,3*j+3)-gt(i,3*j+3);
            sqrt_error(count) = sqrt_error(count)+sqrt(x*x + y*y +z*z);
        
            x=change(i,3*j+1)-gt(i,3*j+1);
            y=change(i,3*j+2)-gt(i,3*j+2);
            z=change(i,3*j+3)-gt(i,3*j+3);
            sqrt_error2(count) = sqrt_error2(count)+sqrt(x*x + y*y +z*z);
        end
    count = count+1;
end
sqrt_error = sqrt_error/num_skel;
sqrt_error2 = sqrt_error2/num_skel;

eval_(folder_i,1)=sum(sqrt_error)/length(protocp2);
eval_(folder_i,2)=sum(sqrt_error2)/length(protocp2);
disp([action_id,':',num2str(length(protocp2)),' eval = ',num2str(eval_(folder_i,1),4),'  change = ',num2str(eval_(folder_i,2),4)]);

end
mean(eval_)


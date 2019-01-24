close all;
clear;clc;
addpaths;

lr = 0.0001;
% root_folder1='/home/cyan/code/tensorflow/z_refine/p1_all/end2end-200000/';
root_folder1='/home/cyan/code/tensorflow/z_refine/humaneva/1213/';
eval=zeros(15);
m=[];
skel = load('skel_16');
skel = skel.skel;

s = '%s%f';
for i=1:80
   s=strcat(s,'%f');
end
ss = '%s%f';
for i=1:31
   ss=strcat(ss,'%f');
end
max_min = csvread('/home/cyan/data/HumanEVA/humaneva_junxin/tools3/point_max_min_16.txt');
% max_min = csvread('/home/cyan/data/human3.6m/annotation/16point_mean_limb_scaled_max_min.csv');
% root_folder = '/home/cyan/data/human3.6m/annotation/16test/';
root_folder = '/home/cyan/data/HumanEVA/new_img_list/3d_16/';
root_2dhg = '/home/cyan/cp_to_shenji/12/humaneva/test2d-150000/';
im_root = '/home/cyan/data/HumanEVA/';%'/home/cyan/data/human3.6m/';

for folder_i = 1:9
    action_id = num2str(folder_i);
    fliename = fopen(strcat(root_folder,'test',action_id,'.txt'));
    file = strcat(num2str(lr),'mpjp_',action_id,'_v2.txt');
    fid = fopen(fullfile(root_folder1, file),'r');
    a = cell2mat(textscan(fid,'%f%f%f%f%f%f'));
    [min_a, I] = min(a,[],2);
    sub_a = a(:,1) - min_a;
    idex = find(sub_a>20);
    m(folder_i,:) = mean(a);
    disp([action_id,' eval = ',num2str(m(folder_i,1)),' ', num2str(m(folder_i,2)),' ',num2str(m(folder_i,3)),' ',num2str(m(folder_i,4)),' ', num2str(m(folder_i,5)),' ',num2str(m(folder_i,6))]);
    
    gt2d3d = textscan(fliename,s,'delimiter',[' ',',']);
    im_name = gt2d3d{1,1};
    gt2d3d{1,1}=[];
    gt2d3d = cell2mat(gt2d3d);
    gt_tmp = gt2d3d(:,33:80);
    gt_tmp = gt_tmp .* repmat(max_min(1,:)-max_min(2,:),[size(gt_tmp,1),1]) + repmat(max_min(2,:),[size(gt_tmp,1),1]);
    
%     hg2d_tmp = textscan(fopen([root_2dhg,'test',action_id,'_square2d.txt'],'r'),ss,'delimiter',[' ',',']);
%     hg2d_tmp{1,1}=[];
%     hg2d_tmp = cell2mat(hg2d_tmp);
    pred2d_tmp = csvread([root_2dhg,'test',action_id,'_norm.csv']);
    hg2d_tmp = pred2d_tmp(:,2:end);

    rpsm_tmp = csvread(['/home/cyan/cp_to_shenji/10/end2end-390000/result',action_id,'_unnorm.csv']);
    rpsm_tmp = rpsm_tmp(:,2:end);
    for j=50:length(idex)
        idex(j)
        im = imread([im_root,im_name{idex(j)}]);
        refine_tmp = csvread([root_folder1,'record_s',num2str(I(idex(j))-1),'_',action_id,'.txt']);
        refine = refine_tmp(2*idex(j),:);
        refine = reshape(refine,3,16);
        rpsm = rpsm_tmp(idex(j),:);
        rpsm = reshape(rpsm,3,16);
        gt = gt_tmp(idex(j),:);
        gt = reshape(gt,3,16);
        hg2d = hg2d_tmp(idex(j),:);
        hg2d = hg2d .* max(size(im));
        hg2d = reshape(hg2d,2,16);
        
        clf;
        figure(1),
        subplot(1,4,1),imshow(im);
        hold on
        vis2Dskel(hg2d,skel);title('im & 2d');
        hold off
        subplot(1,4,2),vis3Dskel(gt*0.003,skel, 'viewpoint',[15 15]);title('gt');
        subplot(1,4,3),vis3Dskel(rpsm*0.003,skel, 'viewpoint',[15 15]);title('rpsm');
        subplot(1,4,4),vis3Dskel(refine*0.003,skel, 'viewpoint',[15 15]);title('refine');
        pause();     
    end  
end

% figure,vis3Dskel(rpsm*0.003,skel, 'viewpoint',[15 15]);title('rpsm');
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_rpsm'),'epsc');
% figure,vis3Dskel(refine*0.003,skel, 'viewpoint',[15 15]);title('refine');
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_refine'),'epsc');
% figure,imshow(im);
% hold on
% vis2Dskel(hg2d,skel);title('im & 2d');
% hold off
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_im'),'epsc');
% figure,vis3Dskel(gt*0.003,skel, 'viewpoint',[15 15]);title('gt');
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_gt'),'epsc');
clear;clc;close all;
load('A_recovered.mat')
load('P_recovered.mat')
load('pupil_recovered.mat')
load('intensity_recovered.mat')
figure;
fsize =10;
set(gcf,'Position',[100 100 400 400]);
subplot(2,2,1)
imshow(squeeze(A),[]);
title('Amplitude','fontsize',fsize)
subplot(2,2,2)
imshow(squeeze(P),[]);
title('Phase','fontsize',fsize)
subplot(2,2,3)
x = squeeze(angle(pupil));
x(x<-0.9*pi) = x(x<-0.9*pi)+2*pi;
imshow((squeeze(x)./pi),[-1,1],'colormap',hsv)
title('Pupil','fontsize',fsize)
subplot(2,2,4)
imshow(intensity,[0,1],'colormap',gray)
title('Intensity distribution','fontsize',fsize)













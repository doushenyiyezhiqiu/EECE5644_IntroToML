x=[-4:4];
y=[4.71738246 , 4.87292099 ,5.0042501  ,5.12729531 ,5.16698752 ,5.17258342,5.17317075, 5.17322979 ,5.17323569];
plot(x,y);
title("MAP model perform on the validation dataset as gamma is varied");
xlabel("covariance matrix hyperparameter gamma");
ylabel("Average squared error on validation dataset");
set(gca,'XTick',-4:4);
set(gca,'XTicklabel',{'10^{-4}','10^{-3}','10^{-2}','10^{-1}','0','10^1','10^2','10^3','10^4'});
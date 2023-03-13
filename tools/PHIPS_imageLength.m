function [field_x_C1,field_y_C1,field_x_C2,field_y_C2,pixel_x_C1,pixel_x_C2] = PHIPS_imageLength(campaign,C1_setting,C2_setting)
% Calculates the image length of C1 and C2
% Resize C2 -> field_x_C1/field_x_C2
% length of a 200 um size bar -> 200/pixel_x_C1

if strcmp(campaign,'SOCRATES') || strcmp(campaign,'CapeEx2019') || strcmp(campaign,'IMPACTS')% Zoom thread not the same as before
    field_diameter_C1 = (1.4/(C1_setting.*2.2)).*7.86;
    field_diameter_C2 = (1.4/(C2_setting.*2)).*7.86;
elseif strcmp(campaign,'ACLOUD') || strcmp(campaign,'ARISTO2017') || strcmp(campaign,'ARISTO2016')
    field_diameter_C1 = (1.4/(C1_setting.*2)).*7.86;
    field_diameter_C2 = (1.4/(C2_setting.*2)).*7.86;
else
    print('Campaign not listed.')
end

% Pixel specifications for GE1380
% Number of pixels in both directions 
x = 1360;
y = 1024; 

% Aspect ratio
ar = x./y;

% Field of view in both directions
field_y_C1 = sqrt(field_diameter_C1.^2./(ar.^2+1));
field_x_C1 = field_y_C1.*ar;
field_y_C2 = sqrt(field_diameter_C2.^2./(ar.^2+1));
field_x_C2 = field_y_C2.*ar;

% Pixel size
pixel_x_C1 = field_x_C1.*1000./x;
pixel_x_C2 = field_x_C2.*1000./x;

disp(['The size of a 200um scale for C1 is ', num2str(200/pixel_x_C1), 'px']) 
disp(['To fit the C1/C2 images you have to increase the size of the C2 image by ', num2str(field_x_C2/field_x_C1 * 100 - 100), '%'])


end


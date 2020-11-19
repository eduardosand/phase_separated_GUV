% Shape of 2 phase vesicle
% Last update: August 15th 2019, Eduardo Sandoval

%% General Explanation
% This script macroscopically generates a phase plane diagram of a phase
% separated vesicle at various pressures and membrane tensions using two
% functions
% More specifically, this script contains two functions.
% Function 1. function [r,h,theta,slope,curvature] = membrane_shape_centraldiff
% contour_length,numberofsteps,phase_boundary_fraction,initial_slope_guess,
% initial_curvature_guess,initial_radius,sigma,kappa,pressure,extrap,
% interpolationnumber
% There are lots of parameters here which I feel are adequately named. The
% functions and where these parameters are defined give more information if
% needed. In essence, this function calculates membrane angle as a function
% of arc length, as well as the first and second derivative using central
% difference formulas.
% We then plug these values into the shape equation, which defines the
% third derivative of membrane angle, or membrane jerk. The shape equation
% can be derived using Euler-Lagrange variational methods and describes an
% equilibrium given force constraints. Due to stability concerns, we start
% the simulation in the middle of the contour when the membrane angle is 
% at pi, simulating up to one end and then back calculating to the 
% other end. Finally, because the vesicle is phase separated, differing
% membrane tensions for each phase results in a different end point radius,
% defined as the shortest distance to the y-axis, or height
% As such, in most cases we have to extrapolate the rest of the vesicle
% Lastly, we also extrapolate since this simulation is a 1D contour
% To obtain a 2D dimensional shape, we assume the axisymmetric case.
%
%
% Function 2. function [] = membrane_shape_phase_plane
% Parameters: contour_length,numberofsteps,phase_boundary_fraction,
% initial_slope_list,initial_curvature_list,initial_radius_list,
% sigma_list,kappa,pressure_list,extrapolationnumbers,baselinesigma,
% baselinepressure,spacing
% As the previous function, there are many parameters. Here the most
% important hard coded parameter is the spacing, which characterizes the
% minimum variations in pressures and membrane tensions we expect to see
% and then generates the phase plane diagram accordingly to ensure the
% vesicles are evenly spaced and labelled accoding to this regimen.
% Again a finer-grained explanation is detailed throughout the function.
% The main idea of this function is to do the same as the first function
% but for a series of solutions and to plot them accordingly all on the
% same figure
% This is done by providing a list of slopes, curvatures, radii, membrane
% tensions for phase 2, pressures, and extrapolation numbers, and the same
% parameters required for function 1. A further idea for improvement is to
% change this to only require a .csv file that has all this information
% already provided in the file.

%% Defining Parameters
clear all
% We use parameters that may be helpful to vary and are therefore defined
% outside of the function

% contour_length defines the length of our simulation in nm. Since we are 
% in the 1D axisymmetric case, the total contour_length is 1/2
% of a vesicle's(after projecting to 2D) total circumference
contour_length = 1000*pi;

% The number of steps defines how many total steps we will take to simulate
% across the contour_length. Together with the contour_length, these two
% parameters define the resolution of our simulation.
numberofsteps = 2000;

% This next parameter defines the fraction of the total that each
% respective phase in the two-phase-separated vesicle will take up. For
% example, if it is switched to 1/3, the first phase will be defined for 1/3
% of the contour with the second phase defining the remaining 2/3 of
% the contour. The excel file provided considers phase boundaries at
% 1/2 of the contour for all cases.
phase_boundary_fraction = 1/2; 

% The vesicle has two phases with potentially varying membrane tension and
% binding modulus. Both are initialized here.
sigma = [1*10^(-13+12),1.*10^(-13+12)]; % pN/nm
kappa = [4000*4, 4000*4]; % pN*nm

% We define pressure this way, because it presupposes a perfectly spherical
% vesicle. If one used a perfect circle and substituted the arc length
% formula into the shape equation, one is able to solve for pressure and
% obtaining the formula found below
pressure = -2*sigma(1)/(contour_length/pi); % pN/nm^2

%% Finding a solution to the shape equation for one set of parameters
% When generating these plots, it's best to set extrapolation to false at
% first, while using the shoot-and-match method to solve, and extrapolating
% only after a solution is found
extrapolation= true;
extrapolationnumber = 10;%55;
% These three must be determined empirically for each set of parameters
% defining a vesicle
% In the perfectly spherical case, the slope is constant and 
% slope = 1/(contour_length/pi) = 1/radius
% Additionally, in the perfectly spherical case, the curvature is 0
% Lastly, since we begin at the maximum radius, the initial radius is
% contour_length/pi, since the contour_length is 1/2 the circumference of
% the vesicle. As a result, our guesses are varied from these starting
% points, regardless of the conditions.
initial_slope_guess = 1/(contour_length/pi);%+29.19065*10^-5;
initial_curvature_guess =0;%45.9*10^-8;
initial_radius = contour_length/pi;%+151;
% This is commented out and is used only when needing to find new solutions
% To find new solutions, guess slopes, curvatures, and radii. Then assess
% radii and theta at either end of the contour. The ideal solution has both
% ends pointed towards the end, at least one end with a radii<1, and thetas
% near 0 and pi depending on the end of the contour.
% [radius_1,height_1,theta_1,slope_1,curvature_1] = membrane_shape_centraldiff(contour_length,numberofsteps,phase_boundary_fraction,initial_slope_guess,initial_curvature_guess,initial_radius,sigma,kappa,pressure,extrapolation,extrapolationnumber);

% new_radius = radius_1(all(radius_1>0,1));
% new_height = height_1(all(radius_1>0,1));
% new_height = new_height-min(new_height);
% contour_length = new_radius(1);
% surface_area = new_radius(1)*2*pi*new_radius(1);
% surface_area2 = new_radius(1)^2*pi;
% for i = 2:length(new_radius)
%     dx = new_radius(i)-new_radius(i-1);
%     dy = new_height(i)-new_height(i-1);
%     contour_length = contour_length + sqrt(dx^2+dy^2);
%     surface_area = surface_area + sqrt(dx^2+dy^2)*2*pi*(new_radius(i)+new_radius(i-1)/2);
%     surface_area2 = surface_area2 + new_radius(i)*sqrt(1+(dy/dx)^2)*dx*2*pi;
% end
% contour_length = contour_length + new_radius(length(new_radius));
% surface_area = surface_area + new_radius(length(new_radius))*2*pi*new_radius(length(new_radius));
% surface_area2 = surface_area2 + new_radius(length(new_radius))^2*pi
% % plot(radius_1,height_1)
% plot(new_height,new_radius)
% axis([-1500,3000,-1500,3000])
% axis square
% % radius_1(length(theta_1))
% % radius_1(1)
% % theta_1(length(theta_1))
% % theta_1(1)
% title('P = 1.2 P_0, \sigma_2 = 1.2 * \sigma_1') 
%% Generating the Phase-Plane Diagram
% Here parameters are read in from an excel file, and then
% used in a function that plots all shapes
% Note that a spacing term is used to denote the difference in relative
% pressure and sigma(0.2). If we wanted to more densely find the shapes(say
% every 0.1, we would need to change this parameter.
parameters = readtable('shape.xlsx');
parameters = table2array(parameters(:,:));
initial_slope_list = parameters(:,1);
initial_curvature_list = parameters(:,2);
sigma_list = parameters(:,3);
pressure_list = parameters(:,4);
baselinesigma = .1
baselinepressure = -2*baselinesigma/(contour_length/pi);
initial_radii_list = parameters(:,5)
extrapolationnumbers = parameters(:,6)
membrane_shape_phase_plane(contour_length,numberofsteps,phase_boundary_fraction,initial_slope_list,initial_curvature_list,initial_radii_list,sigma_list,kappa,pressure_list,extrapolationnumbers,baselinesigma,baselinepressure,0.2)




% Before explaining the function as we go, I will talk about my gripes with
% the function. 
% One is the magic number, which is estimated to be 3000, the
% maximum diameter seen when I originally solved all these vesicles of
% varying membrane tension and pressure. Ideally, we would like to obtain
% this from the list of solutions, but due to differing amounts of 
% extrapolation, the solutions will all have a different size despite the 
% same contour_length. As such, we will likely run into errors if I tried
% to make a matrix of all solutions and finding the maximum difference of
% each solution. I'm sure there's some funtion that must be able to
% interpolate in between smaller solutions and lengthen them to a desired
% length, but I got lazy.
% My second gripe is the spacing parameter. This defines what is the
% minimal proportional variation of pressure or membrane tension used.
% Again, ideally we would like to just obtain this from the list of
% pressures and membrane tensions used, but I got lazy. Since baseline
% membrane tensions and pressure are used, an improvement might be to just
% compare the values of membrane tensions and pressures to the baselines
% and find the minimum proportional difference. With this space parameter,
% I also find issue that the code in its current state requires that the
% spacing between pressures and membrane tensions be similar. I.e. if we
% had minimum proportional changes in pressures as 0.2 but .1 for membrane
% tension, then the phase plane will look more sparse as a result.
function [] = membrane_shape_phase_plane(contour_length,numberofsteps,phase_boundary_fraction,initial_slope_list,initial_curvature_list,initial_radius_list,sigma_list,kappa,pressure_list,extrapolationnumbers,baselinesigma,baselinepressure,spacing)
figure
% initializing our bounds for radius and height in the phase plane, perhaps
% these variables are poorly named but oh well.
maxradius = 0;
minradius = 0;
maxheight = 0;
minheight = 0;
for currentindex = 1:length(pressure_list)
    magicnumber = 3000; % maximum diameter of our solutions, 
    % to space our solutions on the phase plane
    % initializing two phases of membrane tension
    sigma = [baselinesigma, sigma_list(currentindex)];
    % finding a solution for a set of parameters
    [r,h] = membrane_shape_centraldiff(contour_length,numberofsteps,phase_boundary_fraction,initial_slope_list(currentindex),initial_curvature_list(currentindex),initial_radius_list(currentindex),sigma,kappa,pressure_list(currentindex),true,extrapolationnumbers(currentindex));
    % The changeinheight determines how far we should offset the solution
    % on the plot's yaxis. For instance, if our baseline pressure is 0.0002, but 
    % the pressure for this solution is .00028, then we would expect that
    % the changeinheight would be 6000, given that our spacing is 0.2 or
    % 20%.
    changeinheight = (pressure_list(currentindex)-baselinepressure)/baselinepressure/spacing*magicnumber;
    % The changeinradius determines how far we should offset the solution
    % on the plot's xaxis. It is calculated similarly the changeinheight,
    % but is dependent on relative differences in membrane tension.
    changeinradius = (sigma_list(currentindex) -baselinesigma)/baselinesigma/spacing*magicnumber;
    % Now that we've calculated the changeinradius and changeinheight, we
    % offset the solution on both axes
    r = r +changeinradius;
    h = h + changeinheight;
    % These next 4 parameters help us to define the axes and their labels
    % later on by generating an upper and lower bound for the axes ticks.
    maxradius = max(maxradius,changeinradius);
    minradius = min(minradius,changeinradius);
    maxheight = max(maxheight,changeinheight);
    minheight = min(minheight,changeinheight);
    plot(r,h)
    hold on
end
% Here we generated our labels for each axis' ticks. We first generate a
% list that has a tick offset by 3000, or our magic number from our
% minradius to our maxradius. Next, we divide by our magic number, multiply
% by the spacing parameter(which defines proportional differences) and
% offset by 1. It's best to just run this once and see if it makes sense.
% Note that the ticks themselves must be placed at the graphs values, and
% they are then reculated to label them accordingly.
% The same is done for the y-axis
xlist = minradius - magicnumber:magicnumber: maxradius;
xticklist = xlist/magicnumber*spacing+1;
xticks(xlist)
xticklabels(string(xticklist))
ylist = minheight - magicnumber:magicnumber: maxheight+magicnumber;
yticklist = ylist/magicnumber*spacing+1;
yticks(ylist)
yticklabels(string(yticklist))
% It's always good to know latex so I labeled the axes using latex
% interpreter
xlabel('$\displaystyle\frac{\sigma_2}{\sigma_1}$','interpreter','latex')
ylabel('$\displaystyle\frac{P}{P_0}$','interpreter','latex')
title('Phase plane of Membrane Shape')
hold off
end


function [r,h,theta,slope,curvature] = membrane_shape_centraldiff(contour_length,numberofsteps,phase_boundary_fraction,initial_slope_guess,initial_curvature_guess,initial_radius,sigma,kappa,pressure,extrap,interpolationnumber)
% First we define our resolution, at which point our phase boundary occurs
% and initialize all vectors and variables we plan on obtaining
arclengthstep = contour_length/numberofsteps;
phase_boundary = phase_boundary_fraction*contour_length/arclengthstep;
contour_length_vector = linspace(0, contour_length,numberofsteps+1);
membrane_angle = zeros(1,length(contour_length_vector));
membrane_slope = zeros(1,length(contour_length_vector));
membrane_curvature = zeros(1,length(contour_length_vector));
membrane_jerk = zeros(1,length(contour_length_vector));
radius = zeros(1,length(contour_length_vector));
height = zeros(1,length(contour_length_vector));
% We use our initial conditions, as well as our initial guesses to begin
% the simulation. We use these initial conditions to calculate the initial
% membrane_jerk as defined by our equation
% Please note we start our simulation halfway through the contour_length,
% since the simulation is extremely unstable at small radii due to the
% 1/r^3 dependence.
radius(1) = initial_radius;
iterative_radius = initial_radius;
height(1) = 0;
iterative_height = 0;
membrane_angle(1) = pi/2;
membrane_slope(1) = initial_slope_guess;
membrane_curvature(1)= initial_curvature_guess;
% This formula is the shape equation for the 1D axisymmetric vesicle, and 
% therefore isn't evaluated using
% any finite difference formulas.
membrane_jerk(1) = (-1/2 * (membrane_slope(1)^3)) - 2 * ... 
cos(membrane_angle(1))/radius(1)*membrane_curvature(1)+(3/2*...
sin(membrane_angle(1))/radius(1))*(membrane_slope(1)^2)+(3*...
(cos(membrane_angle(1))^2)-1)/(2*radius(1)^2)*membrane_slope(1)-(...
(cos(membrane_angle(1))^2)+1)/(2*(radius(1)^3))*sin(membrane_angle(1))+...
(sigma(1)/kappa(1))*(membrane_slope(1)+sin(membrane_angle(1))/radius(1))+pressure/...
kappa(1);
for i = 2:length(contour_length_vector)
    if i == 2
        % We use a central finite difference method, which results in a
        % modified step for the second time step of simulation
        % We assume membrane_angle(i-2) = membrane_angle(i-1)
        membrane_angle(i) = membrane_angle(i-1) + 2*arclengthstep*membrane_slope(i-1);
        membrane_slope(i) = membrane_slope(i-1) + 2*membrane_curvature(i-1)*arclengthstep;
        membrane_curvature(i) = membrane_curvature(i-1) + 2*membrane_jerk(i-1)*arclengthstep;
        % We update the radius and height for each step, this is estimated
        % by using trigonometry and assuming that our arclengthstep is
        % small enough that this error will not be too far off.
        iterative_radius = iterative_radius + arclengthstep * cos(membrane_angle(i));
        radius(i) = iterative_radius;
        iterative_height = iterative_height + arclengthstep * sin(membrane_angle(i));
        height(i) = iterative_height;
        % Again, we have the shape equation
        membrane_jerk(i) = ((-1/2) * (membrane_slope(i)^3)) - 2 * ... 
                            cos(membrane_angle(i))/radius(i)*membrane_curvature(i)+(3/2*...
                            sin(membrane_angle(i))/radius(i))*(membrane_slope(i)^2)+(3*...
                            (cos(membrane_angle(i))^2)-1)/(2*(radius(i)^2))*membrane_slope(i)-(...
                            (cos(membrane_angle(i))^2)+1)/(2*(radius(i)^3))*sin(membrane_angle(i))+...
                            (sigma(1)/kappa(1))*(membrane_slope(i)+sin(membrane_angle(i))/radius(i))+pressure/...
                            kappa(1);
    elseif i <= int16(numberofsteps/2) + 1
        if i<= int16(phase_boundary) + 1
            membrane_angle(i) = membrane_angle(i-2) + 2*arclengthstep*membrane_slope(i-1);
            % This if loop is purely for debugging and seeing where a
            % simulation breaks to adjust guesses
            if (membrane_angle(i) > (2 * pi) || membrane_angle(i) < - (2*pi))
                i
                membrane_slope(i) = membrane_slope(i-2) + 2*membrane_curvature(i-1)*arclengthstep;
                membrane_curvature(i) = membrane_curvature(i-2) + 2*membrane_jerk(i-1)*arclengthstep;
                iterative_radius = iterative_radius + arclengthstep * cos(membrane_angle(i));
                radius(i) = iterative_radius;
                iterative_height = iterative_height + arclengthstep * sin(membrane_angle(i));
                height(i) = iterative_height;
                membrane_jerk(i) = (-1/2) * (membrane_slope(i)^3) - 2 * ...
                    cos(membrane_angle(i))/radius(i)*membrane_curvature(i)+(3/2*...
                    sin(membrane_angle(i))/radius(i))*(membrane_slope(i)^2)+(3*...
                    (cos(membrane_angle(i))^2)-1)/(2*radius(i)^2)*membrane_slope(i)-(...
                    (cos(membrane_angle(i))^2)+1)/(2*radius(i)^3)*sin(membrane_angle(i))+...
                    (sigma(1)/kappa(1))*(membrane_slope(i)+sin(membrane_angle(i))/radius(i))+pressure/...
                    kappa(1);
                continue
            else
                membrane_slope(i) = membrane_slope(i-2) + 2*membrane_curvature(i-1)*arclengthstep;
                membrane_curvature(i) = membrane_curvature(i-2) + 2*membrane_jerk(i-1)*arclengthstep;
                iterative_radius = iterative_radius + arclengthstep * cos(membrane_angle(i));
                radius(i) = iterative_radius;
                iterative_height = iterative_height + arclengthstep * sin(membrane_angle(i));
                height(i) = iterative_height;
                membrane_jerk(i) = (-1/2) * (membrane_slope(i)^3) - 2 * ...
                    cos(membrane_angle(i))/radius(i)*membrane_curvature(i)+(3/2*...
                    sin(membrane_angle(i))/radius(i))*(membrane_slope(i)^2)+(3*...
                    (cos(membrane_angle(i))^2)-1)/(2*radius(i)^2)*membrane_slope(i)-(...
                    (cos(membrane_angle(i))^2)+1)/(2*radius(i)^3)*sin(membrane_angle(i))+...
                    (sigma(1)/kappa(1))*(membrane_slope(i)+sin(membrane_angle(i))/radius(i))+pressure/...
                    kappa(1);
            end
        elseif i == int16(phase_boundary) + 2
            % This is our first interesting step, as we have reached our
            % phase_boundary or transition. Relevant formulas can be found
            % in Jian Liu's paper: 
            % Endocytic vesicle scission by lipid phase boundary forces
            membrane_angle(i) = membrane_angle(i-2) + 2*arclengthstep*membrane_slope(i-1);
            iterative_radius = iterative_radius + arclengthstep * cos(membrane_angle(i));
            radius(i) = iterative_radius;
            iterative_height = iterative_height + arclengthstep * sin(membrane_angle(i));
            height(i) = iterative_height;
            phase_1_membrane_slope = membrane_slope(i-2) - 2*membrane_curvature(i-1)*arclengthstep;
            phase_1_membrane_curvature = membrane_curvature(i-2) - 2*membrane_jerk(i-1)*arclengthstep;
            phase_2_membrane_slope = (kappa(1) / kappa(2) * phase_1_membrane_slope) - (sin(membrane_angle(i))...
                                  * (kappa(2) - kappa(1))) / (kappa(2) * radius(i));
            membrane_slope(i) = phase_2_membrane_slope;
            phase_2_membrane_curvature = (kappa(1) / kappa(2)) * (phase_1_membrane_curvature -...
                                     (sin(membrane_angle(i)) * ...
                                      cos(membrane_angle(i))) / (radius(i)^2) +...
                                      (cos(membrane_angle(i)) * ...
                                      phase_1_membrane_slope / radius(i))) - ...
                                      (cos(membrane_angle(i)) * membrane_slope(i)) / radius(i) + ...
                                      (sin(membrane_angle(i)) * cos(membrane_angle(i))) / ...
                                      (radius(i)^2);
            membrane_curvature(i) = phase_2_membrane_curvature;
            membrane_jerk(i) = (-1/2) * (membrane_slope(i)^3) - 2 * ... 
                            cos(membrane_angle(i))/radius(i)*membrane_curvature(i)+(3/2*...
                            sin(membrane_angle(i))/radius(i))*(membrane_slope(i)^2)+(3*...
                            (cos(membrane_angle(i))^2)-1)/(2*(radius(i)^2))*membrane_slope(i)-(...
                            (cos(membrane_angle(i))^2)+1)/(2*(radius(i)^3))*sin(membrane_angle(i))+...
                            (sigma(2)/kappa(2))*(membrane_slope(i)+sin(membrane_angle(i))/radius(i))+pressure/...
                            kappa(2);
        else
            membrane_angle(i) = membrane_angle(i-2) + 2*arclengthstep*membrane_slope(i-1);
            membrane_slope(i) = membrane_slope(i-2) + 2*membrane_curvature(i-1)*arclengthstep;
            membrane_curvature(i) = membrane_curvature(i-2) + 2*membrane_jerk(i-1)*arclengthstep;
            iterative_radius = iterative_radius + arclengthstep * cos(membrane_angle(i));
            radius(i) = iterative_radius;
            iterative_height = iterative_height + arclengthstep * sin(membrane_angle(i));
            height(i) = iterative_height;
            membrane_jerk(i) = (-1/2) * (membrane_slope(i)^3) - 2 * ...
                    cos(membrane_angle(i))/radius(i)*membrane_curvature(i)+(3/2*...
                    sin(membrane_angle(i))/radius(i))*(membrane_slope(i)^2)+(3*...
                    (cos(membrane_angle(i))^2)-1)/(2*radius(i)^2)*membrane_slope(i)-(...
                    (cos(membrane_angle(i))^2)+1)/(2*radius(i)^3)*sin(membrane_angle(i))+...
                    (sigma(2)/kappa(2))*(membrane_slope(i)+sin(membrane_angle(i))/radius(i))+pressure/...
                    kappa(2);
        end
    elseif i == int16(numberofsteps/2)+ 2
        % When we have reached halfway through the simulation, we have
        % to back calculate the other half, since we began at half of the
        % contour. We start by initializing a new index to make the
        % computation easier, since now we're backcalculating and will
        % subtract the index by one each step.
        j = int16(numberofsteps/2);
        % We then take our simulation results so far and define them as the
        % second half of the solution. I.e. if we have a 1000 length
        % vector, and got to 500 and now must backcalculate, we redefined
        % the last 500 of the vector to our current results
        membrane_angle(j+1:length(contour_length_vector)) = membrane_angle(1:i-1);
        membrane_curvature(j+1:length(contour_length_vector))= membrane_curvature(1:i-1);
        membrane_slope(j+1:length(contour_length_vector))=membrane_slope(1:i-1);
        membrane_jerk(j+1:length(contour_length_vector))=membrane_jerk(1:i-1);
        % The bottom four lines may not be needed and are only used to
        % placate my paranoia that previously defined values are not
        % refinded and maintain inaccuracies
        membrane_angle(1:j) = 0;
        membrane_jerk(1:j) = 0;
        membrane_curvature(1:j) =0;
        membrane_slope(1:j) = 0;
        height(j+1:length(contour_length_vector)) = height(1:i-1);
        radius(j+1:length(contour_length_vector)) = radius(1:i-1);
        height(1:j) = 0;
        radius(1:j) = 0;
        membrane_angle(j) = membrane_angle(j+2) - 2*arclengthstep*membrane_slope(j+1);
        iterative_radius = initial_radius;
        iterative_radius = iterative_radius - arclengthstep*cos(membrane_angle(j));
        radius(j) = iterative_radius;
        iterative_height = 0;
        iterative_height = iterative_height - arclengthstep*sin(membrane_angle(j));
        height(j) = iterative_height;
        if i == int16(phase_boundary) + 2
            phase_1_membrane_slope = membrane_slope(j+2) - 2*membrane_curvature(j+1)*arclengthstep;
            phase_1_membrane_curvature = membrane_curvature(j+2) - 2*membrane_jerk(j+1)*arclengthstep;
            phase_2_membrane_slope = (kappa(1) / kappa(2) * phase_1_membrane_slope) - (sin(membrane_angle(j))...
                * (kappa(2) - kappa(1))) / (kappa(2) * radius(j));
            membrane_slope(j) = phase_2_membrane_slope;
            phase_2_membrane_curvature = (kappa(1) / kappa(2)) * (phase_1_membrane_curvature -...
                (sin(membrane_angle(j)) * ...
                cos(membrane_angle(j))) / (radius(j)^2) +...
                (cos(membrane_angle(j)) * ...
                phase_1_membrane_slope / radius(j))) - ...
                (cos(membrane_angle(j)) * membrane_slope(j)) / radius(j) + ...
                (sin(membrane_angle(j)) * cos(membrane_angle(j))) / ...
                (radius(j)^2);
            membrane_curvature(j) = phase_2_membrane_curvature;
            % Note that since we are in the second phase, the membrane
            % shape is now using the second phase measurements of membrane
            % tension and binding modulus.
            membrane_jerk(j) = (-1/2) * (membrane_slope(j)^3) - 2 * ...
                cos(membrane_angle(j))/radius(j)*membrane_curvature(j)+(3/2*...
                sin(membrane_angle(j))/radius(j))*(membrane_slope(j)^2)+(3*...
                (cos(membrane_angle(j))^2)-1)/(2*(radius(j)^2))*membrane_slope(j)-(...
                (cos(membrane_angle(j))^2)+1)/(2*(radius(j)^3))*sin(membrane_angle(j))+...
                (sigma(2)/kappa(2))*(membrane_slope(j)+sin(membrane_angle(j))/radius(j))+pressure/...
                kappa(2);
        else
            membrane_slope(j) = membrane_slope(j+2) - 2*membrane_curvature(j+1)*arclengthstep;
            membrane_curvature(j) = membrane_curvature(j+2) - 2*membrane_jerk(j+1)*arclengthstep;
            membrane_jerk(j) = (-1/2) * (membrane_slope(j)^3) - 2 * ... 
                            cos(membrane_angle(j))/radius(j)*membrane_curvature(j)+(3/2*...
                            sin(membrane_angle(j))/radius(j))*(membrane_slope(j)^2)+(3*...
                            (cos(membrane_angle(j))^2)-1)/(2*radius(j)^2)*membrane_slope(j)-(...
                            (cos(membrane_angle(j))^2)+1)/(2*radius(j)^3)*sin(membrane_angle(j))+...
                            (sigma(2)/kappa(2))*(membrane_slope(j)+sin(membrane_angle(j))/radius(j))+pressure/...
                            kappa(2);
        end
    else
        j = j-1;
        membrane_angle(j) = membrane_angle(j+2) - 2*arclengthstep*membrane_slope(j+1);
        if (membrane_angle(j) > (2 * pi) || membrane_angle(j) < - (2*pi))
            i
            break
        else
            iterative_radius = iterative_radius - arclengthstep*cos(membrane_angle(j));
            iterative_height = iterative_height - arclengthstep*sin(membrane_angle(j));
            membrane_slope(j) = membrane_slope(j+2) - 2*membrane_curvature(j+1)*arclengthstep;
            membrane_curvature(j) = membrane_curvature(j+2) - 2*membrane_jerk(j+1)*arclengthstep;
            radius(j) = iterative_radius;
            height(j) = iterative_height;
            membrane_jerk(j) = (-1/2) * (membrane_slope(j)^3) - 2 * ... 
                            cos(membrane_angle(j))/radius(j)*membrane_curvature(j)+(3/2*...
                            sin(membrane_angle(j))/radius(j))*(membrane_slope(j)^2)+(3*...
                            (cos(membrane_angle(j))^2)-1)/(2*radius(j)^2)*membrane_slope(j)-(...
                            (cos(membrane_angle(j))^2)+1)/(2*radius(j)^3)*sin(membrane_angle(j))+...
                            (sigma(2)/kappa(2))*(membrane_slope(j)+sin(membrane_angle(j))/radius(j))+pressure/...
                            kappa(2);
        end
    end
end
r = radius;
h = height;
theta = membrane_angle;
slope = membrane_slope;
curvature= membrane_curvature;
if extrap
    % To extrapolate we first take whatever our extrapolationnumber
    % is(defined as interpolationnumber because when we consider the total
    % vesicle, technically we are interpolating because we are filling in
    % between known points), and find the minimum index for either end of
    % the contour that corresponds to this value
    [~, closestIndex] = min(abs(radius(int16(phase_boundary_fraction*numberofsteps):numberofsteps+1)-interpolationnumber));
    [~, closestIndex_2] = min(abs(radius(1:int16(phase_boundary_fraction*numberofsteps))-interpolationnumber));
    % We use that index to construct a new vector and thus solution that
    % reflects the rest of the solution about the y-axis and generates a
    % vector to fill in the rest
    radius = radius(closestIndex_2:closestIndex+phase_boundary_fraction*numberofsteps);
    height = height(closestIndex_2:closestIndex+phase_boundary_fraction*numberofsteps);
    newradius = zeros(1,2*length(radius));
    newradius(1:length(radius)) = radius;
    newradius(length(radius)+1:2*length(radius)) = -fliplr(radius);
    newheight = zeros(1,2*length(height));
    newheight(1:length(height)) = height;
    newheight(length(height)+1:length(height)*2)= fliplr(height);
    interpolationpoints = -interpolationnumber:arclengthstep:interpolationnumber;
    interpolation = interp1(newradius(length(newradius)/2-interpolationnumber:length(newradius)/2+interpolationnumber),newheight(length(newradius)/2-interpolationnumber:length(newradius)/2+interpolationnumber),-interpolationpoints,'spline');
    interpradius = zeros(1,2*length(radius));
    interpradius(1:length(radius)) = fliplr(radius);
    interpradius(length(radius)+1:2*length(radius)) = -radius;
    interpheight = zeros(1,2*length(radius));
    interpheight(1:length(height)) = fliplr(height);
    interpheight(length(radius)+1:2*length(radius)) = height;
    interpolation2 = interp1(interpradius(length(newradius)/2-interpolationnumber:length(newradius)/2+interpolationnumber),interpheight(length(newradius)/2-interpolationnumber:length(newradius)/2+interpolationnumber),interpolationpoints,'spline');
    newradius = zeros(1,2*length(radius)+2*length(interpolationpoints));
    newradius(1:length(interpolationpoints)) = interpolationpoints;
    newradius(length(interpolationpoints)+1:length(radius)+length(interpolationpoints)) = radius;
    newradius(length(radius)+length(interpolationpoints)+1:length(radius)+2*length(interpolationpoints)) = -interpolationpoints;
    newradius(length(radius)+2*length(interpolationpoints)+1:2*length(radius)+2*length(interpolationpoints)) = -fliplr(radius);
    newheight = zeros(1,2*length(radius)+2*length(interpolationpoints));
    newheight(1:length(interpolationpoints)) = interpolation2;
    newheight(length(interpolationpoints)+1:length(radius)+length(interpolationpoints)) = height;
    newheight(length(radius)+length(interpolationpoints)+1:length(radius)+2*length(interpolationpoints)) = interpolation;
    newheight(length(radius)+2*length(interpolationpoints)+1:2*length(radius)+2*length(interpolationpoints))= fliplr(height);
    r = newradius;
    h = newheight;
else
end
end


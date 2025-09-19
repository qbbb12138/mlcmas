function plot_3d_thermo_umhcolor_roi

csvFile      = 'preproperties_predictions.csv';
outFormation = 'surface3D_formation_umhcolor_roi.png';
outEnthalpy  = 'surface3D_enthalpy_umhcolor_roi.png';

xRange = [2.67 2.69];
yRange = [28.8 29.0];
zRangeForm = [-610 -608];   
zRangeEnth = [];            

useReverse = false;         
showPoints = true;       
viewAZEL   = [45 28];       
lightOn    = true;         
faceAlpha  = 1.0; 

climRange  = [0 7];      
cbTicks    = 0:1:7;       

set(groot,'DefaultAxesFontName','Times New Roman');
set(groot,'DefaultTextFontName','Times New Roman');

T = readWithPreserve(csvFile);
vn = T.Properties.VariableNames;

x  = colD(T,'delta_energy');
y  = colD(T,'delta_modulus_re');
zF = colD(T,'formation');
zH = colD(T,'enthalpy');

try
    C_orig = colD(T,'CVModel_Predicted_crg');
catch
    C_orig = colD(T,'XGB_Predicted_crg');
end

C = min(max(C_orig, climRange(1)), climRange(2));
C(isnan(C)) = climRange(1);   

in = x >= xRange(1) & x <= xRange(2) & y >= yRange(1) & y <= yRange(2);
x = x(in); y = y(in); zF = zF(in); zH = zH(in); C = C(in);
if numel(x) < 3
    error('Too few data points within ROI (<3). Please expand xRange/yRange or check the data.');
end

tri = delaunay(x, y);

cmap = makeRYGB(256);
if useReverse, cmap = flipud(cmap); end

figure('Color','w','Position',[100 100 780 680]);
hs = trisurf(tri, x, y, zF, C);
set(hs,'EdgeColor','none','FaceColor','interp','FaceAlpha',faceAlpha);
xlabel('delta\_energy'); ylabel('delta\_modulus1'); zlabel('formation');
title('3D surface: formation (colored by Predicted umh)');
colormap(cmap);
cb = colorbar; cb.Label.String = 'Predicted umh'; cb.Ticks = cbTicks;
caxis(climRange);                  
xlim(xRange); ylim(yRange);
if ~isempty(zRangeForm), zlim(zRangeForm); end
shading interp; view(viewAZEL); axis tight; box on; grid on;
hold on;
if showPoints
    scatter3(x, y, zF, 18, C, 'filled', 'MarkerEdgeColor','k', 'LineWidth',0.2);
end
hold off;
if lightOn, camlight headlight; material dull; end
saveFigure(outFormation);

figure('Color','w','Position',[920 100 780 680]);
hs2 = trisurf(tri, x, y, zH, C);
set(hs2,'EdgeColor','none','FaceColor','interp','FaceAlpha',faceAlpha);
xlabel('delta\_energy'); ylabel('delta\_modulus1'); zlabel('enthalpy');
title('3D surface: enthalpy (colored by Predicted umh)');
colormap(cmap);
cb = colorbar; cb.Label.String = 'Predicted umh'; cb.Ticks = cbTicks;
caxis(climRange);               
xlim(xRange); ylim(yRange);
if ~isempty(zRangeEnth), zlim(zRangeEnth); end
shading interp; view(viewAZEL); axis tight; box on; grid on;
hold on;
if showPoints
    scatter3(x, y, zH, 18, C, 'filled', 'MarkerEdgeColor','k', 'LineWidth',0.2);
end
hold off;
if lightOn, camlight headlight; material dull; end
saveFigure(outEnthalpy);

fprintf('Saved:\n  %s\n  %s\n', outFormation, outEnthalpy);
end

function T = readWithPreserve(csvFile)
    try
        opts = detectImportOptions(csvFile);
        try, opts.VariableNamingRule = 'preserve'; catch, end
        T = readtable(csvFile, opts);
    catch
        T = readtable(csvFile,'FileType','text','Delimiter',{'\t',','},'ReadVariableNames',true);
    end
end

function v = colD(T, name)
    vn = T.Properties.VariableNames;
    idx = find(strcmpi(vn, name), 1);
    if isempty(idx)
        vnNorm = lower(regexprep(vn,'[^a-z0-9]',''));
        target = lower(regexprep(name,'[^a-z0-9]',''));
        idx = find(strcmp(vnNorm, target), 1);
    end
    if isempty(idx)
        error('Required column not found: %s; available columns: %s', name, strjoin(vn, ', '));
    end
    v = double(T{:, idx});
end

function cmap = makeRYGB(N)
    if nargin < 1, N = 256; end
    stops = [0; 1/3; 2/3; 1];
    cols  = [1 0 0; 1 1 0; 0 1 0; 0 0 1];
    xi = linspace(0,1,N)';
    r = interp1(stops, cols(:,1), xi, 'linear');
    g = interp1(stops, cols(:,2), xi, 'linear');
    b = interp1(stops, cols(:,3), xi, 'linear');
    cmap = [r, g, b];
end

function saveFigure(path)
    if exist('exportgraphics','file')
        exportgraphics(gcf, path, 'Resolution',300);
    else
        set(gcf,'PaperPositionMode','auto');
        print(gcf, path, '-dpng', '-r300');
    end
end

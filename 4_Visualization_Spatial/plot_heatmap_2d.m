function plot_crg_heatmap(fileIn)

    if nargin < 1
        fileIn = 'filtered_preproperties_predictions.csv';
    end

    T = tryReadTableFlexible(fileIn);

    rawNames = T.Properties.VariableNames;

    normNames = lower(regexprep(rawNames, '[^a-zA-Z0-9]+', '_'));
    T.Properties.VariableNames = matlab.lang.makeUniqueStrings(normNames);
    names = T.Properties.VariableNames; 

    xName  = pickFirst(names, {'delta_energy'});  
    yName  = pickFirst(names, {'delta_modulus_re','delta_modulus1','delta_modulus','delta_modulus_reo'});
    crgName = pickFirst(names, {'xgb_predicted_crg','cvmodel_predicted_crg','crg'});

    x = []; y = []; crg = [];

    if ~isempty(xName) && ~isempty(yName) && ~isempty(crgName)
        x   = toNum(T.(xName));
        y   = toNum(T.(yName));
        crg = toNum(T.(crgName));
        fprintf('Using columns: x=%s, y=%s, crg=%s\n', xName, yName, crgName);
    else
        A = buildNumericMatrix(T);
        validCols = find(sum(isfinite(A),1) >= 3); 
        if numel(validCols) >= 3
            x   = A(:, validCols(1));
            y   = A(:, validCols(2));
            crg = A(:, validCols(3));
            fprintf('No matching named columns found, using the first three numeric columns instead.\n');
        else
            need = 'x∈{delta_energy}, y∈{delta_modulus_re|delta_modulus1|delta_modulus|delta_modulus_reo}, crg∈{xgb_predicted_crg|cvmodel_predicted_crg|crg} or 3 numeric columns (x y crg)';
            error('Unrecognized columns: expected %s. Available columns: %s', need, strjoin(rawNames, ', '));
        end
    end

    valid = isfinite(x) & isfinite(y) & isfinite(crg);
    x = x(valid); y = y(valid); crg = crg(valid);
    if numel(x) < 3
        error('Insufficient valid points (more than three required)');
    end

    crg = min(max(crg, 0), 4.5);

    DT  = delaunayTriangulation(x, y);
    tri = DT.ConnectivityList; P = DT.Points;
    figure('Color','w');
    h = trisurf(tri, P(:,1), P(:,2), crg);
    view(2); shading interp; set(h,'EdgeColor','none');
    axis equal tight; box on;
    xlabel('delta\_energy');
    ylabel(strrep(yName,'_','\_'));
    colormap(parula);
    caxis([0 4.5]);
    cb = colorbar; cb.Label.String = 'crg';
    cb.Ticks = 0:0.5:4.5;
    title('CRG Heatmap');

    xlim([2.6805 2.6827]);
    ylim([28.91 28.93]);

    hold on; plot(x, y, '.', 'MarkerSize', 6, 'Color', [0 0 0]); hold off;

    outPng = 'crg_heatmap.png';
    try
        exportgraphics(gcf, outPng, 'Resolution', 300);
    catch
        print(gcf, '-dpng', '-r300', outPng);
    end
    fprintf('Heatmap saved to: %s\n', outPng);
end

function name = pickFirst(names, candidates)
    name = '';
    for i = 1:numel(candidates)
        if ismember(candidates{i}, names)
            name = candidates{i};
            return;
        end
    end
end

function T = tryReadTableFlexible(fileIn)
    try
        opts = detectImportOptions(fileIn, 'FileType','text', ...
            'VariableNamingRule','preserve');
        try opts.Delimiter = {'\t', ',', ';', ' ', '|'}; catch, end
        T = readtable(fileIn, opts, 'TextType','string', ...
            'ReadVariableNames', true, 'MultipleDelimsAsOne', true);
    catch
        T = readtable(fileIn, 'FileType','text', 'TextType','string', ...
            'Delimiter', {'\t', ',', ';', ' ', '|'}, ...
            'ReadVariableNames', true, 'MultipleDelimsAsOne', true, ...
            'VariableNamingRule','preserve');
    end

    if width(T) == 1
        fid = fopen(fileIn, 'r');
        assert(fid>0, 'Unable to open file: %s', fileIn);
        C = textscan(fid, '%s', 1, 'Delimiter', '\n', 'Whitespace','');
        hdr = C{1}{1}; fclose(fid);

        if contains(hdr, sprintf('\t')), delim = '\t';
        elseif contains(hdr, ','),      delim = ',';
        elseif contains(hdr, ';'),      delim = ';';
        elseif contains(hdr, '|'),      delim = '|';
        else,                            delim = ' ';
        end

        T = readtable(fileIn, 'FileType','text', 'TextType','string', ...
            'Delimiter', delim, 'ReadVariableNames', true, ...
            'MultipleDelimsAsOne', true, 'VariableNamingRule','preserve');
    end
end

function v = toNum(col)
    if isnumeric(col), v = double(col);
    else,              v = str2double(string(col));
    end
end

function A = buildNumericMatrix(T)
    names = T.Properties.VariableNames;
    n = height(T); m = width(T);
    A = nan(n, m);
    for j = 1:m
        A(:, j) = toNum(T.(names{j}));
    end
end

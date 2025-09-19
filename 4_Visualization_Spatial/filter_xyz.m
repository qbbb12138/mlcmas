fileIn  = 'preproperties_predictions.csv';
fileOut = 'filtered_preproperties_predictions.csv';

T = readtable(fileIn, 'FileType', 'text', 'TextType', 'string');

origNames = T.Properties.VariableNames;
cleanNames = matlab.lang.makeValidName(strtrim(origNames));
T.Properties.VariableNames = cleanNames;

need = {'delta_energy','delta_modulus_re','formation'};
missing = setdiff(need, T.Properties.VariableNames);
if ~isempty(missing)
    error('Missing required columns: %s\nAvailable columns: %s, ...
        strjoin(missing, ', '), strjoin(T.Properties.VariableNames, ', '));
end

xrange = [2.68, 2.685];     % delta_energy
yrange = [28.9, 29.95];     % delta_modulus1
zrange = [-609, -608.5];     % formation

mask =  T.delta_energy  >= xrange(1) & T.delta_energy  <= xrange(2) & ...
        T.delta_modulus_re>= yrange(1) & T.delta_modulus_re<= yrange(2) & ...
        T.formation     >= zrange(1) & T.formation     <= zrange(2);

T_out = T(mask, :);

writetable(T_out, fileOut);

fprintf('Filtered %d rows of data and saved to: %s\n', height(T_out), fileOut);

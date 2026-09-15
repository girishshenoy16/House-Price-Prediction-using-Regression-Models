let modelData = null;

const DEFAULT_VALUES = {
    overallQual: 6, totalBsmtSF: 1000, firstFlrSF: 1100, secondFlrSF: 0,
    fullBath: 2, halfBath: 1, bsmtFullBath: 0, bsmtHalfBath: 0,
    yearBuilt: 2000, yearRemodAdd: 2000, yrSold: 2008,
    openPorchSF: 30, enclosedPorch: 0, ssnPorch: 0, screenPorch: 0, woodDeckSF: 0,
    msZoning: 'RL', neighborhood: 'NAmes', bldgType: '1Fam', houseStyle: '1Story',
    exterQual: 'TA', kitchenQual: 'TA', foundation: 'PConc', garageType: 'Attchd'
};

const TRAINING_PRICE_MIN = 34900;
const TRAINING_PRICE_MAX = 755000;

// Valuation recommendation thresholds (fraction of training range)
const VAL_LOWER_BOUND = TRAINING_PRICE_MIN + (TRAINING_PRICE_MAX - TRAINING_PRICE_MIN) * 0.20;
const VAL_UPPER_BOUND = TRAINING_PRICE_MIN + (TRAINING_PRICE_MAX - TRAINING_PRICE_MIN) * 0.80;

const NEIGHBORHOOD_LABELS = {
    'NAmes': 'North Ames', 'CollgCr': 'College Creek', 'Gilbert': 'Gilbert',
    'Somerst': 'Somerset', 'NWAmes': 'Northwest Ames', 'Edwards': 'Edwards',
    'BrkSide': 'Brookside', 'OldTown': 'Old Town', 'Sawyer': 'Sawyer',
    'SawyerW': 'Sawyer West', 'NridgHt': 'Northridge Heights',
    'NoRidge': 'Northridge', 'StoneBr': 'Stone Brook', 'Timber': 'Timberland',
    'Crawfor': 'Crawford', 'Mitchel': 'Mitchell', 'IDOTRR': 'Iowa DOT & Rail Road',
    'MeadowV': 'Meadow Village', 'ClearCr': 'Clear Creek', 'Blmngtn': 'Bloomington Heights',
    'BrDale': 'Briardale', 'Veenker': 'Veenker', 'SWISU': 'S&W Iowa State Univ.',
    'Blueste': 'Bluestem', 'NPkVill': 'Northpark Villa'
};

const BUILDING_LABELS = {
    '1Fam': '1-Family', '2fmCon': '2-Family', 'Duplex': 'Duplex',
    'TwnhsE': 'Townhouse End', 'Twnhs': 'Townhouse'
};

const STYLE_LABELS = {
    '1Story': '1 Story', '2Story': '2 Story', '1.5Fin': '1.5 Fin',
    '1.5Unf': '1.5 Unf', 'SFoyer': 'Split Foyer', 'SLvl': 'Split Level'
};

const VALIDATION_RULES = {
    overallQual:     { min: 1, max: 10, label: 'Overall Quality' },
    totalBsmtSF:     { min: 0, label: 'Basement Area' },
    firstFlrSF:      { min: 0, label: '1st Floor Area' },
    secondFlrSF:     { min: 0, label: '2nd Floor Area' },
    fullBath:        { min: 0, max: 5, label: 'Full Bathrooms' },
    halfBath:        { min: 0, max: 3, label: 'Half Bathrooms' },
    bsmtFullBath:    { min: 0, max: 3, label: 'Basement Full Baths' },
    bsmtHalfBath:    { min: 0, max: 2, label: 'Basement Half Baths' },
    yearBuilt:       { min: 1900, max: 2010, label: 'Year Built' },
    yearRemodAdd:    { min: 1900, max: 2010, label: 'Year Remodeled' },
    yrSold:          { min: 2006, max: 2010, label: 'Year Sold' },
    openPorchSF:     { min: 0, label: 'Open Porch' },
    enclosedPorch:   { min: 0, label: 'Enclosed Porch' },
    ssnPorch:        { min: 0, label: '3-Season Porch' },
    screenPorch:     { min: 0, label: 'Screen Porch' },
    woodDeckSF:      { min: 0, label: 'Wood Deck' }
};

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const resp = await fetch('model_export.json');
        modelData = await resp.json();
        document.getElementById('predictBtn').addEventListener('click', onPredictClick);
        document.getElementById('resetBtn').addEventListener('click', resetToDefaults);
        document.getElementById('clearBtn').addEventListener('click', clearInputs);
        document.querySelectorAll('.scenario-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                loadScenario(this.dataset.scenario);
            });
        });
        setupValidation();
        predict();
        renderModelTable();
        renderScatterChart();
        renderFeatureChart();
        renderPriceAreaChart();
    } catch (e) {
        console.error('Failed to load model_export.json:', e);
    }
});

function setupValidation() {
    Object.keys(VALIDATION_RULES).forEach(id => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener('input', () => validateField(id));
        input.addEventListener('blur', () => validateField(id));
    });
    const yearBuilt = document.getElementById('yearBuilt');
    const yrSold = document.getElementById('yrSold');
    if (yearBuilt) yearBuilt.addEventListener('input', () => {
        validateField('yearBuilt');
        validateField('yrSold');
    });
    if (yrSold) yrSold.addEventListener('input', () => validateField('yrSold'));
}

function validateField(id) {
    const input = document.getElementById(id);
    const errEl = document.getElementById('err-' + id);
    if (!input || !errEl) return true;
    const val = parseFloat(input.value);
    const rule = VALIDATION_RULES[id];
    if (isNaN(val)) {
        errEl.textContent = 'Please enter a valid number.';
        input.setAttribute('aria-invalid', 'true');
        return false;
    }
    if (rule.min !== undefined && val < rule.min) {
        errEl.textContent = rule.label + ' must be at least ' + rule.min + '.';
        input.setAttribute('aria-invalid', 'true');
        return false;
    }
    if (rule.max !== undefined && val > rule.max) {
        errEl.textContent = rule.label + ' must be at most ' + rule.max + '.';
        input.setAttribute('aria-invalid', 'true');
        return false;
    }
    if (id === 'yrSold') {
        const built = parseInt(document.getElementById('yearBuilt').value);
        if (!isNaN(built) && val < built) {
            errEl.textContent = 'Year Sold should not be earlier than Year Built.';
            input.setAttribute('aria-invalid', 'true');
            return false;
        }
    }
    errEl.textContent = '';
    input.removeAttribute('aria-invalid');
    return true;
}

function validateAll() {
    let valid = true;
    Object.keys(VALIDATION_RULES).forEach(id => {
        if (!validateField(id)) valid = false;
    });
    return valid;
}

function onPredictClick() {
    if (!validateAll()) return;
    const btn = document.getElementById('predictBtn');
    btn.classList.add('loading');
    btn.setAttribute('aria-busy', 'true');
    setTimeout(() => {
        predict();
        btn.classList.remove('loading');
        btn.removeAttribute('aria-busy');
        showSuccessToast();
    }, 250);
}

function showSuccessToast() {
    const toast = document.getElementById('successToast');
    if (!toast) return;
    toast.style.display = 'block';
    toast.style.animation = 'none';
    toast.offsetHeight;
    toast.style.animation = '';
    setTimeout(() => { toast.style.display = 'none'; }, 2100);
}

function resetToDefaults() {
    Object.entries(DEFAULT_VALUES).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    });
    document.getElementById('overallQualVal').textContent = DEFAULT_VALUES.overallQual;
    clearValidationState();
    clearPredictionState();
    clearActiveScenario();
    predict();
}

function clearInputs() {
    const ids = Object.keys(DEFAULT_VALUES);
    ids.forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        if (el.tagName === 'SELECT') {
            el.selectedIndex = 0;
        } else {
            el.value = '';
        }
    });
    document.getElementById('overallQualVal').textContent = '5';
    clearValidationState();
    clearPredictionState();
    clearActiveScenario();
}

function clearValidationState() {
    Object.keys(VALIDATION_RULES).forEach(id => {
        const errEl = document.getElementById('err-' + id);
        if (errEl) errEl.textContent = '';
        const input = document.getElementById(id);
        if (input) input.removeAttribute('aria-invalid');
    });
    const errorEl = document.getElementById('predictionError');
    if (errorEl) errorEl.style.display = 'none';
    const toast = document.getElementById('successToast');
    if (toast) toast.style.display = 'none';
}

function clearPredictionState() {
    const priceEl = document.getElementById('predictedPrice');
    if (priceEl) priceEl.textContent = '$0';
    const summaryEl = document.getElementById('summaryText');
    if (summaryEl) summaryEl.textContent = '';
    const valEl = document.getElementById('valuationRecommendation');
    if (valEl) valEl.style.display = 'none';
    const warnEl = document.getElementById('extrapolationWarning');
    if (warnEl) warnEl.style.display = 'none';
    ['featTotalSF','featTotalBaths','featAgeAtSale','featRemodAge','featQualArea','featPorch'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = '0';
    });
}

function clearActiveScenario() {
    document.querySelectorAll('.scenario-btn').forEach(btn => btn.classList.remove('active'));
    const toast = document.getElementById('scenarioToast');
    if (toast) toast.style.display = 'none';
}

function showToast(msg, duration) {
    const toast = document.getElementById('scenarioToast');
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, duration || 1800);
}

const SCENARIOS = {
    typical: {
        overallQual: 6, totalBsmtSF: 800, firstFlrSF: 850, secondFlrSF: 800,
        fullBath: 2, halfBath: 1, bsmtFullBath: 1, bsmtHalfBath: 0,
        yearBuilt: 1990, yearRemodAdd: 1995, yrSold: 2008,
        openPorchSF: 30, enclosedPorch: 0, ssnPorch: 0, screenPorch: 0, woodDeckSF: 80,
        overallCond: 5, moSold: 6,
        msZoning: 'RL', lotShape: 'Reg', landContour: 'Lvl', lotConfig: 'Inside',
        neighborhood: 'NAmes', condition1: 'Norm', bldgType: '1Fam', houseStyle: '1Story',
        overallQualCat: '6', overallCondCat: '5',
        roofStyle: 'Gable', exterior1st: 'VinylSd', exterior2nd: 'VinylSd',
        masVnrType: 'None', foundation: 'CBlock', heating: 'GasA', centralAir: 'Y',
        electrical: 'SBrkr', garageType: 'Attchd', garageFinish: 'RFn',
        saleType: 'WD', saleCondition: 'Normal'
    },
    premium: {
        overallQual: 9, totalBsmtSF: 1400, firstFlrSF: 1500, secondFlrSF: 1200,
        fullBath: 3, halfBath: 1, bsmtFullBath: 1, bsmtHalfBath: 0,
        yearBuilt: 2005, yearRemodAdd: 2008, yrSold: 2008,
        openPorchSF: 80, enclosedPorch: 0, ssnPorch: 0, screenPorch: 0, woodDeckSF: 200,
        overallCond: 8, moSold: 6,
        msZoning: 'RL', lotShape: 'Reg', landContour: 'Lvl', lotConfig: 'Corner',
        neighborhood: 'NoRidge', condition1: 'Norm', bldgType: '1Fam', houseStyle: '2Story',
        overallQualCat: '9', overallCondCat: '8',
        roofStyle: 'Hip', exterior1st: 'HdBoard', exterior2nd: 'HdBoard',
        masVnrType: 'BrkFace', foundation: 'PConc', heating: 'GasA', centralAir: 'Y',
        electrical: 'SBrkr', garageType: 'Attchd', garageFinish: 'Fin',
        saleType: 'WD', saleCondition: 'Normal'
    },
    budget: {
        overallQual: 4, totalBsmtSF: 600, firstFlrSF: 700, secondFlrSF: 0,
        fullBath: 1, halfBath: 0, bsmtFullBath: 0, bsmtHalfBath: 0,
        yearBuilt: 1955, yearRemodAdd: 1955, yrSold: 2008,
        openPorchSF: 0, enclosedPorch: 40, ssnPorch: 0, screenPorch: 0, woodDeckSF: 0,
        overallCond: 4, moSold: 3,
        msZoning: 'RM', lotShape: 'Reg', landContour: 'Lvl', lotConfig: 'Inside',
        neighborhood: 'IDOTRR', condition1: 'Norm', bldgType: '1Fam', houseStyle: '1Story',
        overallQualCat: '4', overallCondCat: '4',
        roofStyle: 'Gable', exterior1st: 'AsbShng', exterior2nd: 'AsbShng',
        masVnrType: 'None', foundation: 'CBlock', heating: 'GasA', centralAir: 'N',
        electrical: 'SBrkr', garageType: 'Detchd', garageFinish: 'Unf',
        saleType: 'WD', saleCondition: 'Normal'
    },
    large: {
        overallQual: 7, totalBsmtSF: 1200, firstFlrSF: 1600, secondFlrSF: 1500,
        fullBath: 3, halfBath: 1, bsmtFullBath: 1, bsmtHalfBath: 1,
        yearBuilt: 1995, yearRemodAdd: 2000, yrSold: 2008,
        openPorchSF: 60, enclosedPorch: 0, ssnPorch: 0, screenPorch: 0, woodDeckSF: 150,
        overallCond: 6, moSold: 7,
        msZoning: 'RL', lotShape: 'IR1', landContour: 'Lvl', lotConfig: 'Inside',
        neighborhood: 'Somerst', condition1: 'Norm', bldgType: '1Fam', houseStyle: '2Story',
        overallQualCat: '7', overallCondCat: '6',
        roofStyle: 'Gable', exterior1st: 'VinylSd', exterior2nd: 'VinylSd',
        masVnrType: 'None', foundation: 'PConc', heating: 'GasA', centralAir: 'Y',
        electrical: 'SBrkr', garageType: 'Attchd', garageFinish: 'RFn',
        saleType: 'WD', saleCondition: 'Normal'
    }
};

function loadScenario(name) {
    const s = SCENARIOS[name];
    if (!s) return;
    Object.entries(s).forEach(([id, val]) => {
        const el = document.getElementById(id);
        if (el) el.value = val;
    });
    const qualSlider = document.getElementById('overallQual');
    if (qualSlider) qualSlider.value = s.overallQual;
    document.getElementById('overallQualVal').textContent = s.overallQual;
    clearValidationState();
    clearPredictionState();
    document.querySelectorAll('.scenario-btn').forEach(btn => btn.classList.remove('active'));
    const activeBtn = document.querySelector('.scenario-btn[data-scenario="' + name + '"]');
    if (activeBtn) activeBtn.classList.add('active');
    showToast('Scenario loaded');
}

document.getElementById('overallQual').addEventListener('input', function() {
    document.getElementById('overallQualVal').textContent = this.value;
    this.setAttribute('aria-valuenow', this.value);
});

function engineerFeatures(inputs) {
    const totalSF = inputs.firstFlrSF + inputs.secondFlrSF + inputs.totalBsmtSF;
    const totalBaths = inputs.fullBath + 0.5 * inputs.halfBath + inputs.bsmtFullBath + 0.5 * inputs.bsmtHalfBath;
    const ageAtSale = inputs.yrSold - inputs.yearBuilt;
    const remodAgeAtSale = inputs.yrSold - inputs.yearRemodAdd;
    const qualAreaIndex = inputs.overallQual * totalSF;
    const totalPorchSF = inputs.openPorchSF + inputs.enclosedPorch + inputs.ssnPorch + inputs.screenPorch + inputs.woodDeckSF;
    return { totalSF, totalBaths, ageAtSale, remodAgeAtSale, qualAreaIndex, totalPorchSF };
}

function getInputs() {
    const numCols = modelData.numeric_columns;
    const numImputer = modelData.numeric_imputer_values;
    const numDefaults = {};
    numCols.forEach((col, i) => { numDefaults[col] = numImputer[i]; });

    const catCols = modelData.categorical_columns;
    const catImputer = modelData.categorical_imputer_values;
    const catDefaults = {};
    catCols.forEach(col => { catDefaults[col] = catImputer[col] || ''; });

    return {
        overallQual: parseInt(document.getElementById('overallQual').value),
        totalBsmtSF: parseFloat(document.getElementById('totalBsmtSF').value) || 0,
        firstFlrSF: parseFloat(document.getElementById('firstFlrSF').value) || 0,
        secondFlrSF: parseFloat(document.getElementById('secondFlrSF').value) || 0,
        fullBath: parseInt(document.getElementById('fullBath').value) || 0,
        halfBath: parseInt(document.getElementById('halfBath').value) || 0,
        bsmtFullBath: parseInt(document.getElementById('bsmtFullBath').value) || 0,
        bsmtHalfBath: parseInt(document.getElementById('bsmtHalfBath').value) || 0,
        yearBuilt: parseInt(document.getElementById('yearBuilt').value) || 2000,
        yearRemodAdd: parseInt(document.getElementById('yearRemodAdd').value) || 2000,
        yrSold: parseInt(document.getElementById('yrSold').value) || 2008,
        openPorchSF: parseFloat(document.getElementById('openPorchSF').value) || 0,
        enclosedPorch: parseFloat(document.getElementById('enclosedPorch').value) || 0,
        ssnPorch: parseFloat(document.getElementById('ssnPorch').value) || 0,
        screenPorch: parseFloat(document.getElementById('screenPorch').value) || 0,
        woodDeckSF: parseFloat(document.getElementById('woodDeckSF').value) || 0,
        msZoning: document.getElementById('msZoning').value,
        neighborhood: document.getElementById('neighborhood').value,
        bldgType: document.getElementById('bldgType').value,
        houseStyle: document.getElementById('houseStyle').value,
        exterQual: document.getElementById('exterQual').value,
        kitchenQual: document.getElementById('kitchenQual').value,
        foundation: document.getElementById('foundation').value,
        garageType: document.getElementById('garageType').value,
        MSSubClass: numDefaults['MSSubClass'] || 50,
        LotFrontage: numDefaults['LotFrontage'] || 70,
        LotArea: numDefaults['LotArea'] || 9485,
        OverallCond: numDefaults['OverallCond'] || 5,
        MasVnrArea: numDefaults['MasVnrArea'] || 0,
        BsmtFinSF1: numDefaults['BsmtFinSF1'] || 0,
        BsmtFinSF2: numDefaults['BsmtFinSF2'] || 0,
        BsmtUnfSF: numDefaults['BsmtUnfSF'] || 0,
        LowQualFinSF: numDefaults['LowQualFinSF'] || 0,
        GrLivArea: (parseFloat(document.getElementById('firstFlrSF').value) || 0) + (parseFloat(document.getElementById('secondFlrSF').value) || 0),
        BedroomAbvGr: numDefaults['BedroomAbvGr'] || 3,
        KitchenAbvGr: numDefaults['KitchenAbvGr'] || 1,
        TotRmsAbvGrd: numDefaults['TotRmsAbvGrd'] || 6,
        Fireplaces: numDefaults['Fireplaces'] || 0,
        GarageYrBlt: numDefaults['GarageYrBlt'] || 0,
        GarageCars: numDefaults['GarageCars'] || 0,
        GarageArea: numDefaults['GarageArea'] || 0,
        PoolArea: numDefaults['PoolArea'] || 0,
        MiscVal: numDefaults['MiscVal'] || 0,
        MoSold: numDefaults['MoSold'] || 6,
        Street: catDefaults['Street'] || 'Pave',
        Alley: catDefaults['Alley'] || '',
        LotShape: catDefaults['LotShape'] || 'Reg',
        LandContour: catDefaults['LandContour'] || 'Lvl',
        Utilities: catDefaults['Utilities'] || 'AllPub',
        LotConfig: catDefaults['LotConfig'] || 'Inside',
        LandSlope: catDefaults['LandSlope'] || 'Gtl',
        Condition1: catDefaults['Condition1'] || 'Norm',
        Condition2: catDefaults['Condition2'] || 'Norm',
        RoofStyle: catDefaults['RoofStyle'] || 'Gable',
        RoofMatl: catDefaults['RoofMatl'] || 'CompShg',
        Exterior1st: catDefaults['Exterior1st'] || 'VinylSd',
        Exterior2nd: catDefaults['Exterior2nd'] || 'VinylSd',
        MasVnrType: catDefaults['MasVnrType'] || 'None',
        ExterCond: catDefaults['ExterCond'] || 'TA',
        BsmtQual: catDefaults['BsmtQual'] || 'TA',
        BsmtCond: catDefaults['BsmtCond'] || 'TA',
        BsmtExposure: catDefaults['BsmtExposure'] || 'No',
        BsmtFinType1: catDefaults['BsmtFinType1'] || 'Unf',
        BsmtFinType2: catDefaults['BsmtFinType2'] || 'Unf',
        Heating: catDefaults['Heating'] || 'GasA',
        HeatingQC: catDefaults['HeatingQC'] || 'TA',
        CentralAir: catDefaults['CentralAir'] || 'Y',
        Electrical: catDefaults['Electrical'] || 'SBrkr',
        Functional: catDefaults['Functional'] || 'Typ',
        FireplaceQu: catDefaults['FireplaceQu'] || '',
        GarageFinish: catDefaults['GarageFinish'] || 'Unf',
        GarageQual: catDefaults['GarageQual'] || 'TA',
        GarageCond: catDefaults['GarageCond'] || 'TA',
        PavedDrive: catDefaults['PavedDrive'] || 'Y',
        PoolQC: catDefaults['PoolQC'] || '',
        Fence: catDefaults['Fence'] || '',
        MiscFeature: catDefaults['MiscFeature'] || '',
        SaleType: catDefaults['SaleType'] || 'WD',
        SaleCondition: catDefaults['SaleCondition'] || 'Normal'
    };
}

function buildPropertySummary(inputs) {
    const totalSF = inputs.firstFlrSF + inputs.secondFlrSF + inputs.totalBsmtSF;
    const baths = inputs.fullBath + inputs.halfBath * 0.5;
    const bathsStr = baths === Math.floor(baths) ? baths.toString() : baths.toFixed(1);
    const hoodLabel = NEIGHBORHOOD_LABELS[inputs.neighborhood] || inputs.neighborhood;
    const bldgLabel = BUILDING_LABELS[inputs.bldgType] || inputs.bldgType;
    return '<strong>' + totalSF.toLocaleString() + ' sqft</strong> &middot; ' +
           bathsStr + ' bath' + (baths !== 1 ? 's' : '') + ' &middot; ' +
           'Built ' + inputs.yearBuilt + ' &middot; ' +
           hoodLabel + ' &middot; ' + bldgLabel;
}

function predict() {
    if (!modelData) return;
    const inputs = getInputs();
    const features = engineerFeatures(inputs);

    document.getElementById('featTotalSF').textContent = features.totalSF.toLocaleString();
    document.getElementById('featTotalBaths').textContent = features.totalBaths.toFixed(1);
    document.getElementById('featAgeAtSale').textContent = features.ageAtSale;
    document.getElementById('featRemodAge').textContent = features.remodAgeAtSale;
    document.getElementById('featQualArea').textContent = features.qualAreaIndex.toLocaleString();
    document.getElementById('featPorch').textContent = features.totalPorchSF.toLocaleString();

    const summaryEl = document.getElementById('summaryText');
    if (summaryEl) summaryEl.innerHTML = buildPropertySummary(inputs);

    const rawInput = { ...inputs, ...features };
    const dollar = computePrediction(modelData, rawInput);
    document.getElementById('predictedPrice').textContent = '$' + Math.round(dollar).toLocaleString();

    const warning = document.getElementById('extrapolationWarning');
    const errorEl = document.getElementById('predictionError');
    if (errorEl) errorEl.style.display = 'none';
    if (warning) {
        if (dollar < TRAINING_PRICE_MIN || dollar > TRAINING_PRICE_MAX) {
            warning.style.display = 'block';
        } else {
            warning.style.display = 'none';
        }
    }

    // Valuation recommendation
    const recEl = document.getElementById('valuationRecommendation');
    const recMsg = document.getElementById('valuationMessage');
    if (recEl && recMsg) {
        if (dollar < TRAINING_PRICE_MIN || dollar > TRAINING_PRICE_MAX) {
            recEl.style.display = 'block';
            recMsg.innerHTML = '<span class="valuation-status caution">Model Extrapolation</span>' +
                'Estimated value: <strong>$' + Math.round(dollar).toLocaleString() + '</strong>. ' +
                'This estimate is outside the observed training range. Treat it as a directional estimate and compare it with similar properties.';
        } else if (dollar < VAL_LOWER_BOUND) {
            recEl.style.display = 'block';
            recMsg.innerHTML = '<span class="valuation-status neutral">Lower Range</span>' +
                'Estimated value: <strong>$' + Math.round(dollar).toLocaleString() + '</strong>. ' +
                'This estimate is near the lower end of the observed training range. Compare it with similar homes with similar size, quality, and condition.';
        } else if (dollar > VAL_UPPER_BOUND) {
            recEl.style.display = 'block';
            recMsg.innerHTML = '<span class="valuation-status neutral">Upper Range</span>' +
                'Estimated value: <strong>$' + Math.round(dollar).toLocaleString() + '</strong>. ' +
                'This estimate is near the upper end of the observed training range. Compare it with similar properties to validate.';
        } else {
            recEl.style.display = 'block';
            recMsg.innerHTML = '<span class="valuation-status neutral">Within Range</span>' +
                'Estimated value: <strong>$' + Math.round(dollar).toLocaleString() + '</strong>. ' +
                'Use this as a reference point when comparing similar homes.';
        }
    }
}

function computePrediction(model, rawInput) {
    const numCols = model.numeric_columns;
    const catCols = model.categorical_columns;
    const numImputer = model.numeric_imputer_values;
    const scalerMean = model.scaler_mean;
    const scalerScale = model.scaler_scale;
    const catImputer = model.categorical_imputer_values;
    const catMaps = model.categorical_maps;
    const coefficients = model.coefficients;
    const intercept = model.intercept;

    const camelToModel = {
        overallQual: 'OverallQual', totalBsmtSF: 'TotalBsmtSF',
        firstFlrSF: '1stFlrSF', secondFlrSF: '2ndFlrSF',
        fullBath: 'FullBath', halfBath: 'HalfBath',
        bsmtFullBath: 'BsmtFullBath', bsmtHalfBath: 'BsmtHalfBath',
        yearBuilt: 'YearBuilt', yearRemodAdd: 'YearRemodAdd',
        yrSold: 'YrSold', openPorchSF: 'OpenPorchSF',
        enclosedPorch: 'EnclosedPorch', ssnPorch: '3SsnPorch',
        screenPorch: 'ScreenPorch', woodDeckSF: 'WoodDeckSF',
        msZoning: 'MSZoning', neighborhood: 'Neighborhood',
        bldgType: 'BldgType', houseStyle: 'HouseStyle',
        exterQual: 'ExterQual', kitchenQual: 'KitchenQual',
        foundation: 'Foundation', garageType: 'GarageType',
    };
    const resolve = (col) => {
        if (rawInput[col] !== undefined) return rawInput[col];
        const camel = Object.keys(camelToModel).find(k => camelToModel[k] === col);
        if (camel && rawInput[camel] !== undefined) return rawInput[camel];
        const generic = col.charAt(0).toLowerCase() + col.slice(1);
        if (rawInput[generic] !== undefined) return rawInput[generic];
        return undefined;
    };

    const numericVals = numCols.map((col, i) => {
        let val = resolve(col);
        if (val === undefined || val === null || isNaN(val)) val = numImputer[i];
        return val;
    });
    const numericScaled = numericVals.map((v, i) => (v - scalerMean[i]) / scalerScale[i]);

    const catVec = [];
    catCols.forEach(col => {
        let val = resolve(col);
        if (val === undefined || val === null || val === '') val = catImputer[col];
        const categories = catMaps[col];
        categories.forEach(cat => {
            catVec.push(val === cat ? 1.0 : 0.0);
        });
    });

    const fullVector = numericScaled.concat(catVec);
    let dotProduct = 0;
    for (let i = 0; i < fullVector.length; i++) {
        dotProduct += fullVector[i] * coefficients[i];
    }
    const innerLog = dotProduct + intercept;

    if (model.target_log_transformed) {
        return Math.expm1(innerLog);
    }
    return innerLog;
}

function formatFeatureName(name) {
    const neighborhoodMap = {
        'NAmes': 'North Ames', 'CollgCr': 'College Creek', 'Gilbert': 'Gilbert',
        'Somerst': 'Somerset', 'NWAmes': 'Northwest Ames', 'Edwards': 'Edwards',
        'BrkSide': 'Brookside', 'OldTown': 'Old Town', 'Sawyer': 'Sawyer',
        'SawyerW': 'Sawyer West', 'NridgHt': 'Northridge Heights',
        'NoRidge': 'Northridge', 'StoneBr': 'Stone Brook', 'Timber': 'Timberland',
        'Crawfor': 'Crawford', 'Mitchel': 'Mitchell', 'IDOTRR': 'Iowa DOT & Rail Road',
        'MeadowV': 'Meadow Village', 'ClearCr': 'Clear Creek', 'Blmngtn': 'Bloomington Heights',
        'BrDale': 'Briardale', 'Veenker': 'Veenker', 'SWISU': 'S&W Iowa State Univ.',
        'Blueste': 'Bluestem', 'NPkVill': 'Northpark Villa'
    };
    const zoningMap = {
        'RL': 'Residential Low', 'RM': 'Residential Medium',
        'FV': 'Floating Village', 'RH': 'Residential High',
        'C': 'Commercial', 'C (all)': 'Commercial'
    };

    if (name.startsWith('Neighborhood_')) {
        const key = name.replace('Neighborhood_', '');
        return 'Neighborhood: ' + (neighborhoodMap[key] || key);
    }
    if (name.startsWith('MSZoning_')) {
        const key = name.replace('MSZoning_', '');
        return 'Zoning: ' + (zoningMap[key] || key);
    }
    if (name.startsWith('BldgType_')) return 'Building: ' + name.replace('BldgType_', '');
    if (name.startsWith('HouseStyle_')) return 'Style: ' + name.replace('HouseStyle_', '');
    if (name.startsWith('ExterQual_')) return 'Ext. Quality: ' + name.replace('ExterQual_', '');
    if (name.startsWith('KitchenQual_')) return 'Kitchen Quality: ' + name.replace('KitchenQual_', '');
    if (name.startsWith('Foundation_')) return 'Foundation: ' + name.replace('Foundation_', '');
    if (name.startsWith('GarageType_')) return 'Garage: ' + name.replace('GarageType_', '');
    if (name.startsWith('HeatingQC_')) return 'Heating Quality: ' + name.replace('HeatingQC_', '');
    if (name.startsWith('CentralAir_')) return 'Central Air: ' + name.replace('CentralAir_', '');
    if (name.startsWith('Electrical_')) return 'Electrical: ' + name.replace('Electrical_', '');
    if (name.startsWith('Functional_')) return 'Functionality: ' + name.replace('Functional_', '');
    if (name.startsWith('FireplaceQu_')) return 'Fireplace: ' + name.replace('FireplaceQu_', '');
    if (name.startsWith('GarageFinish_')) return 'Garage Finish: ' + name.replace('GarageFinish_', '');
    if (name.startsWith('GarageQual_')) return 'Garage Quality: ' + name.replace('GarageQual_', '');
    if (name.startsWith('GarageCond_')) return 'Garage Condition: ' + name.replace('GarageCond_', '');
    if (name.startsWith('PavedDrive_')) return 'Paved Drive: ' + name.replace('PavedDrive_', '');
    if (name.startsWith('SaleType_')) return 'Sale Type: ' + name.replace('SaleType_', '');
    if (name.startsWith('SaleCondition_')) return 'Sale Condition: ' + name.replace('SaleCondition_', '');
    if (name.startsWith('RoofStyle_')) return 'Roof: ' + name.replace('RoofStyle_', '');
    if (name.startsWith('RoofMatl_')) return 'Roof Material: ' + name.replace('RoofMatl_', '');
    if (name.startsWith('Exterior1st_')) return 'Exterior: ' + name.replace('Exterior1st_', '');
    if (name.startsWith('MasVnrType_')) return 'Masonry: ' + name.replace('MasVnrType_', '');
    if (name.startsWith('BsmtQual_')) return 'Basement Quality: ' + name.replace('BsmtQual_', '');
    if (name.startsWith('BsmtCond_')) return 'Basement Cond.: ' + name.replace('BsmtCond_', '');
    if (name.startsWith('BsmtExposure_')) return 'Basement Exposure: ' + name.replace('BsmtExposure_', '');
    if (name.startsWith('BsmtFinType1_')) return 'Basement Finish: ' + name.replace('BsmtFinType1_', '');
    return name;
}

function linearRegression(points) {
    const n = points.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (const p of points) {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumX2 += p.x * p.x;
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    return { slope, intercept };
}

function renderModelTable() {
    if (!modelData) return;
    const cv = modelData.cv_results || {};
    const ho = modelData.holdout_results || {};
    const tbody = document.querySelector('#modelTable tbody');
    const models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Random Forest', 'XGBoost'];
    const selected = modelData.model_name;

    models.forEach(name => {
        const c = cv[name] || {};
        const h = ho[name] || {};
        const tr = document.createElement('tr');
        if (name === selected) tr.classList.add('selected');
        const isSelected = name === selected;
        tr.innerHTML =
            '<td>' + name + (isSelected ? ' <span class="selected-badge">Selected</span>' : '') + '</td>' +
            '<td>$' + Math.round(c.cv_rmse || 0).toLocaleString() + '</td>' +
            '<td>$' + Math.round(c.cv_mae || 0).toLocaleString() + '</td>' +
            '<td>' + (c.cv_r2 || 0).toFixed(4) + '</td>' +
            '<td>$' + Math.round(h.holdout_rmse || 0).toLocaleString() + '</td>' +
            '<td>$' + Math.round(h.holdout_mae || 0).toLocaleString() + '</td>' +
            '<td>' + (h.holdout_r2 || 0).toFixed(4) + '</td>';
        tbody.appendChild(tr);
    });
}

function renderScatterChart() {
    const ctx = document.getElementById('scatterChart').getContext('2d');
    const points = (modelData.holdout_predictions || []).map(function(p) { return { x: p.Actual, y: p.Predicted }; });

    const allVals = points.reduce(function(acc, p) { acc.push(p.x, p.y); return acc; }, []);
    var maxVal = Math.max.apply(null, allVals) * 1.05;

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Holdout Predictions \u2014 Ridge Regression (292 homes)',
                data: points,
                backgroundColor: 'rgba(37,99,235,0.45)',
                borderColor: 'rgba(37,99,235,0.8)',
                pointRadius: 3.5,
                pointHoverRadius: 6,
            }, {
                label: 'Perfect Prediction',
                data: [{x: 0, y: 0}, {x: maxVal, y: maxVal}],
                type: 'line',
                borderColor: 'rgba(239,68,68,0.6)',
                borderDash: [6, 3],
                pointRadius: 0,
                borderWidth: 2,
                fill: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.3,
            plugins: {
                legend: {
                    labels: { color: '#475569', font: { size: 11 }, usePointStyle: true, pointStyle: 'circle' }
                },
                tooltip: {
                    backgroundColor: '#0F172A',
                    titleColor: '#F8FAFC',
                    bodyColor: '#CBD5E1',
                    borderColor: '#334155',
                    borderWidth: 1,
                    cornerRadius: 6,
                    padding: 10,
                    callbacks: {
                        label: function(ctx) {
                            if (ctx.datasetIndex === 0) {
                                var actual = Math.round(ctx.parsed.x);
                                var predicted = Math.round(ctx.parsed.y);
                                var error = Math.abs(actual - predicted);
                                return 'Actual: $' + actual.toLocaleString() +
                                       ' | Predicted: $' + predicted.toLocaleString() +
                                       ' | Error: $' + error.toLocaleString();
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Actual Sale Price ($)', color: '#475569', font: { size: 12, weight: '600' } },
                    ticks: { color: '#94A3B8', font: { size: 10 }, callback: function(v) { return '$' + (v/1000).toFixed(0) + 'k'; } },
                    grid: { color: '#F1F5F9' },
                    min: 0, max: maxVal
                },
                y: {
                    title: { display: true, text: 'Predicted Sale Price ($)', color: '#475569', font: { size: 12, weight: '600' } },
                    ticks: { color: '#94A3B8', font: { size: 10 }, callback: function(v) { return '$' + (v/1000).toFixed(0) + 'k'; } },
                    grid: { color: '#F1F5F9' },
                    min: 0, max: maxVal
                }
            }
        }
    });
}

function renderFeatureChart() {
    if (!modelData || !modelData.coefficients) return;
    var coefs = modelData.coefficients;
    var names = modelData.feature_names;
    var pairs = names.map(function(n, i) { return { name: formatFeatureName(n), imp: Math.abs(coefs[i]) }; });
    pairs.sort(function(a, b) { return b.imp - a.imp; });
    var top = pairs.slice(0, 12).reverse();

    var ctx = document.getElementById('featureChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: top.map(function(p) { return p.name; }),
            datasets: [{
                label: 'Absolute Coefficient',
                data: top.map(function(p) { return p.imp; }),
                backgroundColor: 'rgba(37,99,235,0.7)',
                borderColor: 'rgba(37,99,235,1)',
                borderWidth: 1,
                borderRadius: 3,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.2,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: { color: '#94A3B8', font: { size: 10 } },
                    grid: { color: '#F1F5F9' }
                },
                y: {
                    ticks: { color: '#475569', font: { size: 10 } },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderPriceAreaChart() {
    if (!modelData) return;
    var rawData = modelData.raw_data_sample || [];
    if (rawData.length === 0) return;

    var points = rawData.map(function(d) { return { x: d.GrLivArea, y: d.SalePrice }; });
    var lr = linearRegression(points);

    var minArea = 0;
    var maxArea = Math.max.apply(null, points.map(function(p) { return p.x; })) * 1.05;
    var trendLine = [
        { x: minArea, y: lr.slope * minArea + lr.intercept },
        { x: maxArea, y: lr.slope * maxArea + lr.intercept }
    ];

    var maxPrice = Math.max.apply(null, points.map(function(p) { return p.y; })) * 1.05;

    var ctx = document.getElementById('priceAreaChart').getContext('2d');
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Properties',
                data: points,
                backgroundColor: 'rgba(16,185,129,0.35)',
                borderColor: 'rgba(16,185,129,0.7)',
                pointRadius: 3,
                pointHoverRadius: 5,
            }, {
                label: 'Trend Line',
                data: trendLine,
                type: 'line',
                borderColor: 'rgba(245,158,11,0.7)',
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 1.3,
            plugins: {
                legend: {
                    labels: { color: '#475569', font: { size: 11 }, usePointStyle: true, pointStyle: 'circle' }
                },
                tooltip: {
                    backgroundColor: '#0F172A',
                    titleColor: '#F8FAFC',
                    bodyColor: '#CBD5E1',
                    borderColor: '#334155',
                    borderWidth: 1,
                    cornerRadius: 6,
                    padding: 10,
                    filter: function(tooltipItem) {
                        return tooltipItem.datasetIndex === 0;
                    },
                    callbacks: {
                        label: function(ctx) {
                            return 'Living Area: ' + ctx.parsed.x.toLocaleString() + ' sqft | Sale Price: $' + ctx.parsed.y.toLocaleString();
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Above Ground Living Area (GrLivArea, sqft)', color: '#475569', font: { size: 12, weight: '600' } },
                    ticks: { color: '#94A3B8', font: { size: 10 }, callback: function(v) { return v.toLocaleString(); } },
                    grid: { color: '#F1F5F9' },
                    min: 0, max: maxArea
                },
                y: {
                    title: { display: true, text: 'Sale Price ($)', color: '#475569', font: { size: 12, weight: '600' } },
                    ticks: { color: '#94A3B8', font: { size: 10 }, callback: function(v) { return '$' + (v/1000).toFixed(0) + 'k'; } },
                    grid: { color: '#F1F5F9' },
                    min: 0, max: maxPrice
                }
            }
        }
    });
}

document.getElementById('overallQual').addEventListener('input', function() {
    document.getElementById('overallQualVal').textContent = this.value;
    this.setAttribute('aria-valuenow', this.value);
});

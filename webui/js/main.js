// Конфигурация
const config = {
    airflowUrl: 'http://localhost:8080',
    mlflowUrl: 'http://localhost:5050',
    refreshInterval: 30000 // 30 секунд
};

// Управление параметрами алгоритмов
document.addEventListener('DOMContentLoaded', function() {
    // PCA
    const pcaCheckbox = document.getElementById('pcaCheckbox');
    const pcaParameters = document.getElementById('pcaParameters');
    const pcaInput = document.getElementById('pcaComponents');

    // UMAP
    const umapCheckbox = document.getElementById('umapCheckbox');
    const umapParameters = document.getElementById('umapParameters');

    // Обработчики для PCA
    if (pcaCheckbox && pcaParameters) {
        pcaCheckbox.addEventListener('change', function(e) {
            pcaParameters.style.display = this.checked ? 'block' : 'none';
            if (pcaInput) pcaInput.disabled = this.checked;
        });

        if (pcaInput) {
            setupNumericValidation(pcaInput, 1, 1000);
        }
    }

    // Обработчики для UMAP
    if (umapCheckbox && umapParameters) {
        umapCheckbox.addEventListener('change', function(e) {
            umapParameters.style.display = this.checked ? 'block' : 'none';

            // Disable/Enable все поля UMAP
            const inputs = umapParameters.querySelectorAll('input, select');
            inputs.forEach(input => input.disabled = this.checked);
        });

        // Настройка валидации для числовых полей UMAP
        setupUMAPValidation();
    }
});

// Функция для настройки валидации числовых полей
function setupNumericValidation(input, min, max) {
    input.addEventListener('input', function(e) {
        let value = parseFloat(this.value);

        if (isNaN(value)) {
            this.value = min;
            return;
        }

        if (value < min) this.value = min;
        if (value > max) this.value = max;
    });

    input.addEventListener('keydown', function(e) {
        // Разрешаем специальные клавиши
        if ([46, 8, 9, 27, 13, 37, 38, 39, 40].indexOf(e.keyCode) !== -1 ||
            (e.keyCode === 65 && (e.ctrlKey === true || e.metaKey === true)) ||
            (e.keyCode === 67 && (e.ctrlKey === true || e.metaKey === true)) ||
            (e.keyCode === 86 && (e.ctrlKey === true || e.metaKey === true)) ||
            (e.keyCode === 88 && (e.ctrlKey === true || e.metaKey === true))) {
            return;
        }

        // Запрещаем ввод не цифр
        if ((e.shiftKey || (e.keyCode < 48 || e.keyCode > 57)) &&
            (e.keyCode < 96 || e.keyCode > 105)) {
            e.preventDefault();
        }
    });
}

// Настройка валидации для UMAP полей
function setupUMAPValidation() {
    // n_neighbors
    const neighbors = document.getElementById('umapNeighbors');
    if (neighbors) setupNumericValidation(neighbors, 2, 200);

    // min_dist
    const minDist = document.getElementById('umapMinDist');
    if (minDist) {
        minDist.addEventListener('input', function(e) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = 0.1;
                return;
            }
            if (value < 0) this.value = 0;
            if (value > 0.99) this.value = 0.99;
            this.value = Math.round(this.value * 100) / 100;
        });
    }

    // n_components
    const components = document.getElementById('umapComponents');
    if (components) setupNumericValidation(components, 1, 100);

    // spread
    const spread = document.getElementById('umapSpread');
    if (spread) {
        spread.addEventListener('input', function(e) {
            let value = parseFloat(this.value);
            if (isNaN(value)) {
                this.value = 1.0;
                return;
            }
            if (value < 0.5) this.value = 0.5;
            if (value > 5.0) this.value = 5.0;
            this.value = Math.round(this.value * 10) / 10;
        });
    }
}

// Функция для получения параметров UMAP
function getUMAPParameters() {
    const umapCheckbox = document.getElementById('umapCheckbox');

    if (!umapCheckbox || !umapCheckbox.checked) {
        return { enabled: false };
    }

    return {
        enabled: true,
        n_neighbors: parseInt(document.getElementById('umapNeighbors')?.value || 15),
        min_dist: parseFloat(document.getElementById('umapMinDist')?.value || 0.1),
        n_components: parseInt(document.getElementById('umapComponents')?.value || 2),
        metric: document.getElementById('umapMetric')?.value || 'euclidean',
        spread: parseFloat(document.getElementById('umapSpread')?.value || 1.0),
        low_memory: document.getElementById('umapLowMemory')?.value === 'true',
        init: document.getElementById('umapInit')?.value || 'spectral'
    };
}

// Функция для получения параметров PCA
function getPCAParameters() {
    const pcaCheckbox = document.getElementById('pcaCheckbox');

    if (!pcaCheckbox || !pcaCheckbox.checked) {
        return { enabled: false };
    }

    return {
        enabled: true,
        n_components: parseInt(document.getElementById('pcaComponents')?.value || 2)
    };
}

// Функция для получения всех параметров
function getAllParameters() {
    return {
        pca: getPCAParameters(),
        umap: getUMAPParameters()
    };
}

// Обновленная функция triggerDAG с поддержкой обоих алгоритмов
async function triggerDAG(dagId) {
    const params = getAllParameters();

    try {
        const response = await fetch(`${config.airflowUrl}/api/v1/dags/${dagId}/dagRuns`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                conf: params
            })
        });

        if (response.ok) {
            alert(`DAG ${dagId} успешно запущен с выбранными параметрами`);
        } else {
            alert(`Ошибка при запуске DAG ${dagId}`);
        }
    } catch (error) {
        console.error('Error triggering DAG:', error);
        alert('Ошибка при запуске DAG');
    }
}
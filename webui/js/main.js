document.addEventListener('DOMContentLoaded', function() {
    const pcaCheckbox = document.getElementById('pcaCheckbox');
    const umapCheckbox = document.getElementById('umapCheckbox');

    //  Обработка нажатия чекбокса PCA (inputs enable)
    pcaCheckbox.addEventListener('change', function(event) {
        if (this.checked) {
            console.log('PCA активирован');
            document.getElementById('pcaComponents').disabled = false;
        } else {
            console.log('PCA деактивирован');
            document.getElementById('pcaComponents').disabled = true;
        }
    });

    //  Обработка нажатия чекбокса UMAP (inputs enable)
    umapCheckbox.addEventListener('change', function(event) {
        if (this.checked) {
            console.log('UMAP активирован');
            document.getElementById('umapNeighbors').disabled = false;
            document.getElementById('umapMinDist').disabled = false;
            document.getElementById('umapComponents').disabled = false;
            document.getElementById('umapMetric').disabled = false;
            document.getElementById('umapSpread').disabled = false;
            document.getElementById('umapLowMemory').disabled = false;
            document.getElementById('umapInit').disabled = false;
        } else {
            console.log('UMAP деактивирован');
            document.getElementById('umapNeighbors').disabled = true;
            document.getElementById('umapMinDist').disabled = true;
            document.getElementById('umapComponents').disabled = true;
            document.getElementById('umapMetric').disabled = true;
            document.getElementById('umapSpread').disabled = true;
            document.getElementById('umapLowMemory').disabled = true;
            document.getElementById('umapInit').disabled = true;
        }
    });
});

function refreshData() {
    // Функция для очистки полей пользовательского ввода
    document.getElementById('pcaComponents').value = '';
    document.getElementById('umapNeighbors').value = '';
    document.getElementById('umapMinDist').value = '';
    document.getElementById('umapComponents').value = '';
    document.getElementById('umapMetric').value = '';
    document.getElementById('umapSpread').value = '';
    document.getElementById('umapLowMemory').value = '';
    document.getElementById('umapInit').value = '';
}

function startPipeline() {
    // Функция для запуска Airflow DAG с параметрами алгоритмов, введенными пользователем
    if (checkCorrectInputs()) {
        console.log("Все супер! Запускаем пайплайн!");
    }
    else {
        console.log("Все печально( запустить не получится");
    }
}

function checkCorrectInputs() {
    // Функция для проверки корректности введенных пользователем значений
    if (checkUMAPInputs() && checkPCAInputs()) {
        return true;
    }
    else {
        return false;
    }
}

function checkPCAInputs() {
    // Получаем pcaComponents
    const pcaComponents = document.getElementById('pcaComponents');

    let isValid = true;
    let errors = [];

    if (pcaComponents) {
        const val = parseInt(pcaComponents.value);
        if (isNaN(val) || val < 2) {
            pcaComponents.classList.add('error');
            errors.push('pcaComponents должно быть ≥ 2');
            isValid = false;
        } else if (val > 200) {
            pcaComponents.classList.add('error');
            errors.push('pcaComponents не должно превышать 200');
            isValid = false;
        } else {
            pcaComponents.classList.remove('error');
        }
    }

    // Выводим ошибки если есть
    if (errors.length > 0) {
        alert('Ошибка в параметрах PCA:\n\n' + errors.join('\n'));

        console.warn('Найдены ошибки в параметрах PCA:');
        errors.forEach(err => console.warn('  - ' + err));
    } else {
        console.log('Все параметры PCA корректны');
    }

    return isValid;
}

function checkUMAPInputs() {
    // Получаем все элементы
    const neighbors = document.getElementById('umapNeighbors');
    const minDist = document.getElementById('umapMinDist');
    const components = document.getElementById('umapComponents');
    const metric = document.getElementById('umapMetric');
    const spread = document.getElementById('umapSpread');
    const lowMemory = document.getElementById('umapLowMemory');
    const init = document.getElementById('umapInit');

    let isValid = true;
    let errors = [];

    if (neighbors) {
        const val = parseInt(neighbors.value);
        if (isNaN(val) || val < 2) {
            neighbors.classList.add('error');
            errors.push('n_neighbors должно быть ≥ 2');
            isValid = false;
        } else if (val > 200) {
            neighbors.classList.add('error');
            errors.push('n_neighbors не должно превышать 200');
            isValid = false;
        } else {
            neighbors.classList.remove('error');
        }
    }

    if (minDist) {
        const val = parseFloat(minDist.value);
        if (isNaN(val) || val < 0.0) {
            minDist.classList.add('error');
            errors.push('min_dist должно быть ≥ 0.0');
            isValid = false;
        } else if (val > 0.99) {
            minDist.classList.add('error');
            errors.push('min_dist должно быть ≤ 0.99');
            isValid = false;
        } else {
            minDist.classList.remove('error');
        }
    }

    if (components) {
        const val = parseInt(components.value);
        if (isNaN(val) || val < 1) {
            components.classList.add('error');
            errors.push('n_components должно быть ≥ 1');
            isValid = false;
        } else if (val > 100) {
            components.classList.add('error');
            errors.push('n_components не должно превышать 100');
            isValid = false;
        } else {
            components.classList.remove('error');
        }
    }

    if (spread) {
        const val = parseFloat(spread.value);
        if (isNaN(val) || val < 0.1) {
            spread.classList.add('error');
            errors.push('spread должно быть ≥ 0.1');
            isValid = false;
        } else if (val > 100) {
            spread.classList.add('error');
            errors.push('spread не должно превышать 100');
            isValid = false;
        } else {
            spread.classList.remove('error');
        }
    }

    // Выводим ошибки если есть
    if (errors.length > 0) {
        alert('Ошибки в параметрах UMAP:\n\n' + errors.join('\n'));

        console.warn('Найдены ошибки в параметрах UMAP:');
        errors.forEach(err => console.warn('  - ' + err));
    } else {
        console.log('Все параметры UMAP корректны');
    }

    return isValid;
}
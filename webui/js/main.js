// Конфигурация
const config = {
    airflowUrl: 'http://localhost:8080',
    mlflowUrl: 'http://localhost:5050',
    refreshInterval: 30000 // 30 секунд
};

// Состояние приложения
let appState = {
    airflowStatus: false,
    mlflowStatus: false,
    postgresStatus: false,
    dags: [],
    experiments: []
};

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
    checkServices();
    loadDAGs();
    loadMLflowExperiments();

    // Автоматическое обновление
    setInterval(() => {
        checkServices();
        loadDAGs();
        loadMLflowExperiments();
    }, config.refreshInterval);
});

// Проверка статуса сервисов
async function checkServices() {
    // Проверка Airflow
    try {
        const airflowResponse = await fetch(`${config.airflowUrl}/api/v1/health`, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        appState.airflowStatus = airflowResponse.ok;
    } catch (error) {
        appState.airflowStatus = false;
    }

    // Проверка MLflow
    try {
        const mlflowResponse = await fetch(`${config.mlflowUrl}/api/2.0/mlflow/experiments/list`);
        appState.mlflowStatus = mlflowResponse.ok;
    } catch (error) {
        appState.mlflowStatus = false;
    }

    // Проверка PostgreSQL (через API или просто индикация)
    appState.postgresStatus = appState.airflowStatus; // Если Airflow работает, то и PostgreSQL скорее всего тоже

    updateStatusIndicators();
}

// Обновление индикаторов статуса
function updateStatusIndicators() {
    // Общий статус системы
    const systemStatus = document.getElementById('systemStatus');
    const statusText = document.getElementById('statusText');

    const allServicesOk = appState.airflowStatus && appState.mlflowStatus && appState.postgresStatus;

    systemStatus.className = `status-dot ${allServicesOk ? 'active' : 'inactive'}`;
    statusText.textContent = allServicesOk ? 'Все системы работают' : 'Некоторые сервисы недоступны';

    // Статус отдельных сервисов
    updateServiceStatus('airflowStatus', appState.airflowStatus);
    updateServiceStatus('mlflowStatus', appState.mlflowStatus);
    updateServiceStatus('postgresStatus', appState.postgresStatus);
}

function updateServiceStatus(elementId, isActive) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = isActive ? 'Активен' : 'Недоступен';
        element.className = `stat-value ${isActive ? 'success' : 'error'}`;
    }
}

// Загрузка DAG'ов из Airflow
async function loadDAGs() {
    if (!appState.airflowStatus) {
        document.getElementById('dagsContainer').innerHTML =
            '<div class="error-message">Airflow недоступен</div>';
        return;
    }

    try {
        const response = await fetch(`${config.airflowUrl}/api/v1/dags`, {
            headers: {
                'Accept': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            appState.dags = data.dags;
            renderDAGs();
        }
    } catch (error) {
        console.error('Error loading DAGs:', error);
    }
}

// Отрисовка DAG'ов
function renderDAGs() {
    const container = document.getElementById('dagsContainer');

    if (appState.dags.length === 0) {
        container.innerHTML = '<div class="loading">Нет доступных DAG\'ов</div>';
        return;
    }

    container.innerHTML = appState.dags.map(dag => `
        <div class="dag-card ${dag.is_paused ? 'paused' : 'running'}">
            <div class="dag-name">${dag.dag_id}</div>
            <div class="dag-status">${dag.is_paused ? 'Приостановлен' : 'Активен'}</div>
            <div class="dag-meta">
                <span>Схема: ${dag.schedule_interval || 'Нет'}</span>
                <span>Теги: ${dag.tags?.map(t => t.name).join(', ') || 'нет'}</span>
            </div>
        </div>
    `).join('');
}

// Загрузка экспериментов из MLflow
async function loadMLflowExperiments() {
    if (!appState.mlflowStatus) {
        const tbody = document.getElementById('mlflowBody');
        tbody.innerHTML = '<tr><td colspan="4" class="error-message">MLflow недоступен</td></tr>';
        return;
    }

    try {
        const response = await fetch(`${config.mlflowUrl}/api/2.0/mlflow/experiments/list`);

        if (response.ok) {
            const data = await response.json();
            appState.experiments = data.experiments || [];
            renderMLflowExperiments();
        }
    } catch (error) {
        console.error('Error loading MLflow experiments:', error);
    }
}

// Отрисовка MLflow экспериментов
function renderMLflowExperiments() {
    const tbody = document.getElementById('mlflowBody');

    if (appState.experiments.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="loading">Нет экспериментов</td></tr>';
        return;
    }

    // Берем последние 5 экспериментов
    const recentExperiments = appState.experiments.slice(0, 5);

    tbody.innerHTML = recentExperiments.map(exp => `
        <tr>
            <td>${exp.name}</td>
            <td><span class="status-dot ${exp.lifecycle_stage === 'active' ? 'active' : 'inactive'}" style="display: inline-block; margin-right: 0.5rem;"></span>${exp.lifecycle_stage}</td>
            <td>${exp.artifact_location || 'Н/Д'}</td>
            <td>${new Date(exp.creation_time).toLocaleString()}</td>
        </tr>
    `).join('');
}

// Запуск DAG
async function triggerDAG(dagId) {
    if (!appState.airflowStatus) {
        alert('Airflow недоступен');
        return;
    }

    try {
        const response = await fetch(`${config.airflowUrl}/api/v1/dags/${dagId}/dagRuns`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                conf: {}
            })
        });

        if (response.ok) {
            alert(`DAG ${dagId} успешно запущен`);
        } else {
            alert(`Ошибка при запуске DAG ${dagId}`);
        }
    } catch (error) {
        console.error('Error triggering DAG:', error);
        alert('Ошибка при запуске DAG');
    }
}

// Обновление данных
function refreshData() {
    checkServices();
    loadDAGs();
    loadMLflowExperiments();
}
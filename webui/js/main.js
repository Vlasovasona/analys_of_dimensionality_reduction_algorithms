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
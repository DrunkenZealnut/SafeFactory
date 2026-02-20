/**
 * Common JavaScript functions shared across templates
 * - XSS prevention with escapeHtml
 * - Markdown rendering with renderMarkdown
 * - Image lightbox functionality
 * - Chart rendering with Chart.js
 * - Image gallery rendering
 */

// ============================================================================
// XSS Prevention
// ============================================================================

/**
 * Escape HTML to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} - HTML-escaped text
 */
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Markdown Rendering
// ============================================================================

/**
 * Render Markdown to sanitized HTML with fallback to escaped text.
 * Requires marked.js and DOMPurify to be loaded for Markdown rendering.
 * @param {string} text - Raw Markdown text
 * @returns {string} - Sanitized HTML string
 */
function renderMarkdown(text) {
    if (typeof marked !== 'undefined' && typeof DOMPurify !== 'undefined') {
        var html = marked.parse(text || '');
        return DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
    }
    if (typeof marked !== 'undefined') {
        console.warn('DOMPurify not loaded, markdown disabled for security');
    }
    return escapeHtml(text || '').replace(/\n/g, '<br>');
}

// ============================================================================
// Image Lightbox
// ============================================================================

/**
 * Open lightbox with image
 * @param {string} imageSrc - Image source URL
 */
function openLightbox(imageSrc) {
    const lightbox = document.getElementById('lightbox');
    const lightboxImage = document.getElementById('lightboxImage');
    if (lightbox && lightboxImage) {
        lightboxImage.src = imageSrc;
        lightbox.classList.add('active');
    }
}

/**
 * Close lightbox
 */
function closeLightbox() {
    const lightbox = document.getElementById('lightbox');
    if (lightbox) {
        lightbox.classList.remove('active');
    }
}

/**
 * Initialize lightbox event listeners
 * Must be called after DOM is loaded
 */
function initLightbox() {
    // Image gallery event delegation (XSS-safe)
    document.addEventListener('click', (e) => {
        const imageItem = e.target.closest('.image-item');
        if (imageItem && imageItem.dataset.imagePath) {
            openLightbox(imageItem.dataset.imagePath);
        }
    });

    // Close lightbox on background click
    const lightbox = document.getElementById('lightbox');
    if (lightbox) {
        lightbox.addEventListener('click', (e) => {
            if (e.target.id === 'lightbox') {
                closeLightbox();
            }
        });
    }

    // Close lightbox on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeLightbox();
        }
    });
}

// ============================================================================
// Chart Rendering
// ============================================================================

/**
 * Extract chart data from AI response
 * @param {string} answer - AI response text with embedded chart data
 * @returns {{cleanAnswer: string, charts: Array}} - Clean answer and chart data array
 */
function extractChartData(answer) {
    const chartRegex = /<!--CHART_DATA\s*([\s\S]*?)\s*CHART_DATA-->/g;
    const charts = [];
    let match;

    while ((match = chartRegex.exec(answer)) !== null) {
        try {
            charts.push(JSON.parse(match[1]));
        } catch (e) {
            console.error('Failed to parse chart data:', e);
        }
    }

    // Remove chart data from answer for display
    const cleanAnswer = answer.replace(chartRegex, '');
    return { cleanAnswer, charts };
}

/**
 * Render chart using Chart.js
 * @param {Object} chartData - Chart configuration object
 * @param {string} containerId - Canvas element ID
 * @returns {Chart} - Chart.js instance
 */
function renderChart(chartData, containerId) {
    const canvas = document.getElementById(containerId);
    if (!canvas) {
        console.error(`Canvas element not found: ${containerId}`);
        return null;
    }

    const ctx = canvas.getContext('2d');

    const colors = [
        'rgba(0, 212, 255, 0.8)',
        'rgba(156, 39, 176, 0.8)',
        'rgba(76, 175, 80, 0.8)',
        'rgba(255, 193, 7, 0.8)',
        'rgba(244, 67, 54, 0.8)',
        'rgba(33, 150, 243, 0.8)'
    ];

    const config = {
        type: chartData.type || 'bar',
        data: {
            labels: chartData.labels || [],
            datasets: [{
                label: chartData.title || 'Data',
                data: chartData.data || [],
                backgroundColor: chartData.type === 'line'
                    ? 'rgba(0, 212, 255, 0.2)'
                    : colors.slice(0, chartData.data?.length || 1),
                borderColor: chartData.type === 'line'
                    ? 'rgba(0, 212, 255, 1)'
                    : colors.slice(0, chartData.data?.length || 1),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#e0e0e0' }
                },
                title: {
                    display: true,
                    text: chartData.title || '',
                    color: '#00d4ff',
                    font: { size: 14 }
                }
            },
            scales: chartData.type !== 'pie' ? {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#aaa' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                x: {
                    ticks: { color: '#aaa' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            } : {}
        }
    };

    const chart = new Chart(ctx, config);

    // Store chart instance for cleanup (if activeCharts array exists)
    if (typeof activeCharts !== 'undefined') {
        activeCharts.push(chart);
    }

    return chart;
}

// ============================================================================
// Image Gallery
// ============================================================================

/**
 * Render image gallery HTML
 * @param {Array} images - Array of image objects with path and name
 * @returns {string} - HTML string for image gallery
 */
function renderImageGallery(images) {
    if (!images || images.length === 0) return '';

    let html = `
        <div class="image-gallery">
            <div class="image-gallery-header">
                <span>🖼️</span> 관련 이미지 (${images.length}개) - 클릭하여 확대
            </div>
            <div class="image-grid">
    `;

    images.forEach(img => {
        const escapedPath = escapeHtml(img.path);
        const escapedName = escapeHtml(img.name);
        const displayName = img.name.length > 30
            ? img.name.substring(0, 27) + '...'
            : img.name;
        const escapedDisplayName = escapeHtml(displayName);

        html += `
            <div class="image-item" data-image-path="${escapedPath}">
                <img src="${escapedPath}" alt="${escapedName}" loading="lazy">
                <div class="image-overlay">${escapedDisplayName}</div>
            </div>
        `;
    });

    html += '</div></div>';
    return html;
}

// ============================================================================
// Initialization
// ============================================================================

// Auto-initialize lightbox when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initLightbox);
} else {
    // DOM is already loaded
    initLightbox();
}

console.log('✅ Common.js loaded successfully');

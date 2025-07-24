/**
 * Scroll Progress Indicator
 * This script adds a visual indicator of scroll progress through the content
 */
document.addEventListener('DOMContentLoaded', function() {
    // Create the scroll indicator element
    const scrollIndicator = document.createElement('div');
    scrollIndicator.className = 'scroll-progress-indicator';
    document.body.appendChild(scrollIndicator);
    
    // Update the indicator width on scroll
    window.addEventListener('scroll', function() {
        const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (window.scrollY / windowHeight) * 100;
        scrollIndicator.style.width = scrolled + '%';
    });
});

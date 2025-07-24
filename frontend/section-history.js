/**
 * Section Navigation History
 * This script tracks navigation between sections to enable "back" functionality
 */
document.addEventListener('DOMContentLoaded', function() {
    // Keep track of section navigation history
    const navigationHistory = [];
    let currentSectionIndex = -1;
    
    // Function to navigate back to the previous section
    function navigateToPreviousSection() {
        if (currentSectionIndex > 0) {
            currentSectionIndex--;
            const previousSection = navigationHistory[currentSectionIndex];
            
            // Use existing scrollToSection function but don't record in history
            if (typeof scrollToSection === 'function') {
                const skipHistory = true;
                scrollToSection(previousSection, skipHistory);
            }
        }
    }
    
    // Extend the scrollToSection function to record history
    const originalScrollToSection = window.scrollToSection;
    
    if (typeof originalScrollToSection === 'function') {
        window.scrollToSection = function(elementId, skipHistory) {
            // Call the original function
            originalScrollToSection(elementId);
            
            // Record in history unless specified to skip
            if (!skipHistory) {
                // If we've navigated back and then to a new section, truncate the forward history
                if (currentSectionIndex >= 0 && currentSectionIndex < navigationHistory.length - 1) {
                    navigationHistory.splice(currentSectionIndex + 1);
                }
                
                // Add to history if it's different from current
                if (navigationHistory.length === 0 || 
                    navigationHistory[navigationHistory.length - 1] !== elementId) {
                    navigationHistory.push(elementId);
                    currentSectionIndex = navigationHistory.length - 1;
                }
            }
        };
    }
    
    // Create back navigation buttons where needed
    document.querySelectorAll('.result-card, .analysis-card').forEach(card => {
        // Only add if the card doesn't have navigation buttons already
        const existingControls = card.querySelector('.card-controls');
        if (existingControls) {
            // Check for existing jump buttons
            const existingJump = existingControls.querySelector('.section-jump');
            
            if (!existingJump) {
                // Create new section jump container
                const sectionJump = document.createElement('div');
                sectionJump.className = 'section-jump';
                
                // Create back button
                const backBtn = document.createElement('button');
                backBtn.className = 'jump-btn';
                backBtn.title = 'Back to Previous Section';
                backBtn.innerHTML = '<i class="fas fa-arrow-left"></i>';
                backBtn.addEventListener('click', navigateToPreviousSection);
                
                // Add to the DOM
                sectionJump.appendChild(backBtn);
                existingControls.appendChild(sectionJump);
            } else {
                // Add back button to existing jump container
                const backBtn = document.createElement('button');
                backBtn.className = 'jump-btn';
                backBtn.title = 'Back to Previous Section';
                backBtn.innerHTML = '<i class="fas fa-arrow-left"></i>';
                backBtn.addEventListener('click', navigateToPreviousSection);
                
                // Insert at the beginning
                existingJump.insertBefore(backBtn, existingJump.firstChild);
            }
        }
    });
});

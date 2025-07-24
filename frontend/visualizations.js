// Visualizations for word analysis
document.addEventListener('DOMContentLoaded', function() {
    // This function is now disabled as per request to remove chart visualizations
    window.visualizeWordData = function(elementId, data) {
        // Functionality disabled - charts removed
        return;
        
        // The original implementation is commented out below
        /*
        const container = document.getElementById(elementId);
        if (!container || !data || data.length === 0) return;
        
        // Create a canvas element for visualization
        const canvas = document.createElement('canvas');
        canvas.width = container.clientWidth;
        canvas.height = 200;
        canvas.className = 'word-chart';
        
        // Add canvas before the word items
        const firstWordItem = container.querySelector('.word-item');
        if (firstWordItem) {
            container.insertBefore(canvas, firstWordItem);
        } else {
            container.appendChild(canvas);
        }
        
        // Get the 2d context
        const ctx = canvas.getContext('2d');
        
        // Prepare data
        const labels = data.map(item => item.word);
        const counts = data.map(item => item.count);
        const maxCount = Math.max(...counts);
        
        // Draw bars
        const barWidth = (canvas.width - 40) / data.length;
        const barSpacing = 5;
        const maxHeight = canvas.height - 60;
        
        // Draw title
        ctx.fillStyle = '#333';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Word Frequency', canvas.width / 2, 20);
        
        // Draw bars and labels
        for (let i = 0; i < data.length; i++) {
            const x = 20 + i * (barWidth + barSpacing);
            const barHeight = (data[i].count / maxCount) * maxHeight;
            const y = canvas.height - barHeight - 30;
            
            // Draw gradient bar
            const gradient = ctx.createLinearGradient(x, y, x, canvas.height - 30);
            gradient.addColorStop(0, '#667eea');
            gradient.addColorStop(1, '#764ba2');
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth, barHeight);
            
            // Draw word label
            ctx.fillStyle = '#333';
            ctx.font = '11px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(data[i].word, x + barWidth / 2, canvas.height - 15);
            
            // Draw count
            ctx.fillStyle = '#333';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(data[i].count, x + barWidth / 2, y - 5);
        }
        */
    };
    
    // Rest of the document event listener implementation
});

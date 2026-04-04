import { buildLayout } from "./layout";

export default async function main(game) {
    const container = buildLayout(game.app);
    
    game.stage.aim.visible = false;

    // Run inference requests explicitly via HTTP Fetch instead of Web Workers
    setInterval(async () => {
        const canvas = game.app.renderer.extract.canvas(game.stage);
        
        // Convert canvas strictly to Base64 PEG data URL with 60% quality 
        // to minimize network transmission footprint over localhost.
        const base64Image = canvas.toDataURL('image/jpeg', 0.6);

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Image })
            });

            if (!response.ok) return;

            const data = await response.json();
            
            // Loop predictions precisely mapped to the original worker messages
            for (const pred of data.predictions) {
                console.log(`🎯 AI predicted at: (${pred.x}, ${pred.y})`);
                container.updateHUD(pred);
                game.stage.aim.visible = true;

                game.stage.aim.setPosition(pred.x, pred.y);
                const position = game.stage.aim.getGlobalPosition();

                game.handleClick({
                    global: position,
                });
            }
        } catch (error) {
            console.error("Machine Learning Server might be offline:", error);
        }

    }, 200); // Execute sequentially looping loosely every 200ms

    return container;
}

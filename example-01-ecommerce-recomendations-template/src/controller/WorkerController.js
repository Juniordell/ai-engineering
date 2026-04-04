import { workerEvents } from "../events/constants.js";

export class WorkerController {
    #events;
    #alreadyTrained = false;
    constructor({ worker, events }) {
        this.#events = events;
        this.#alreadyTrained = false;
        this.init();
    }

    async init() {
        this.setupCallbacks();
    }

    static init(deps) {
        return new WorkerController(deps);
    }

    setupCallbacks() {
        this.#events.onTrainModel((data) => {
            this.#alreadyTrained = false;
            this.triggerTrain(data);
        });
        
        this.#events.onRecommend((data) => {
            if (!this.#alreadyTrained) return;
            this.triggerRecommend(data);
        });
    }

    async triggerTrain(users) {
        try {
            // Simulate progression so UI shows activity
            this.#events.dispatchProgressUpdate({ progress: 10 });
            
            const response = await fetch('http://localhost:8000/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ users })
            });
            await response.json();
            
            this.#events.dispatchProgressUpdate({ progress: 100 });
            this.#alreadyTrained = true;
            this.#events.dispatchTrainingComplete({});
        } catch(err) {
            console.error("Training error fetching API", err);
        }
    }

    async triggerRecommend(user) {
        try {
            const response = await fetch('http://localhost:8000/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user })
            });
            const data = await response.json();
            
            this.#events.dispatchRecommendationsReady({
                user: user,
                recommendations: data.recommendations
            });
        } catch(err) {
            console.error("Recommend error fetching API", err);
        }
    }
}
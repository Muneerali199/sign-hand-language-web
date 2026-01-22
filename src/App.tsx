import { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { CheckCircle, Hand, Loader2, Play, Square, AlertTriangle } from 'lucide-react';

const SignLabels = [
  'Hello', 'Thank You', 'Yes', 'No', 'I Love You',
  'Please', 'Sorry', 'Help', 'Good Morning', 'Good Night',
  'Excuse Me', 'How are you', 'Nice to meet you', 'Goodbye', 'See you later'
];

function App() {
  const webcamRef = useRef<Webcam>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [model, setModel] = useState<any | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isDetecting, setIsDetecting] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Load Model
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log('Loading TFLite model...');
        // Wait for tflite to be available on window
        const tflite = (window as any).tflite;
        const tf = (window as any).tf;
        if (!tflite || !tf) {
           throw new Error("TensorFlow.js scripts not loaded");
        }

        // Set WASM path manually if needed
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');

        // Try to load model from public/model.tflite
        const loadedModel = await tflite.loadTFLiteModel('/model.tflite');
        setModel(loadedModel);
        
        // Log model details
        console.log('Model loaded successfully');
        console.log('Inputs:', loadedModel.inputs);
        console.log('Outputs:', loadedModel.outputs);
        
      } catch (error) {
        console.warn('Failed to load model.tflite, falling back to simulation mode.', error);
        setModel({ simulated: true });
        setError('Model file not found. Running in demo mode.');
        setTimeout(() => setError(null), 5000);
      } finally {
        setIsModelLoading(false);
      }
    };

    setTimeout(loadModel, 1000);
  }, []);

  const runInference = useCallback(async () => {
    if (!isDetecting || !webcamRef.current?.video || !model) return;

    if (model.simulated) {
      // Simulation logic
      if (Math.random() > 0.95) {
        const randomIdx = Math.floor(Math.random() * SignLabels.length);
        setPrediction(SignLabels[randomIdx]);
        setConfidence(0.85 + Math.random() * 0.14);
      }
    } else {
      // Real Inference Logic
      try {
        const tf = (window as any).tf;
        const video = webcamRef.current.video;

        if (video.readyState === 4) {
           // 1. Get image from webcam
           const img = tf.browser.fromPixels(video);
           
           // 2. Preprocess (resize to 224x224 and normalize)
           // Adjust size based on your model's requirement
           const resized = tf.image.resizeBilinear(img, [224, 224]);
           const normalized = resized.div(255.0); // Normalize to [0, 1]
           const expanded = normalized.expandDims(0); // Add batch dimension [1, 224, 224, 3]

           // 3. Run inference
           // Note: predict might return a single tensor or an object/array of tensors
           const outputTensor = model.predict(expanded);
           
           // 4. Parse output
           // Assuming output is a classification probability vector
           const data = await outputTensor.data();
           
           // Find max confidence
           let maxScore = -1;
           let maxIndex = -1;
           for (let i = 0; i < data.length; i++) {
             if (data[i] > maxScore) {
               maxScore = data[i];
               maxIndex = i;
             }
           }

           if (maxScore > 0.5 && maxIndex < SignLabels.length) {
             setPrediction(SignLabels[maxIndex]);
             setConfidence(maxScore);
           }

           // Cleanup tensors
           tf.dispose([img, resized, normalized, expanded, outputTensor]);
        }
      } catch (e) {
        console.error("Inference error:", e);
        setIsDetecting(false);
        setError("Inference failed. Check console for details.");
      }
    }

    if (isDetecting) {
       requestAnimationFrame(runInference);
    }
  }, [isDetecting, model]);

  useEffect(() => {
    if (isDetecting) {
      runInference();
    }
  }, [isDetecting, runInference]);

  const toggleDetection = () => {
    setIsDetecting(!isDetecting);
    if (!isDetecting) {
        setPrediction(null);
        setConfidence(0);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md bg-gray-800 rounded-2xl overflow-hidden shadow-xl border border-gray-700">
        {/* Header */}
        <div className="p-4 bg-gray-800 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Hand className="w-6 h-6 text-blue-400" />
            <h1 className="text-xl font-bold">Signify Web</h1>
          </div>
          {isModelLoading ? (
            <div className="flex items-center gap-2 text-sm text-yellow-400">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading Model...</span>
            </div>
          ) : (
            <div className="flex items-center gap-2 text-sm text-green-400">
              <CheckCircle className="w-4 h-4" />
              <span>{model?.simulated ? 'Demo Mode' : 'Model Ready'}</span>
            </div>
          )}
        </div>

        {/* Camera Feed */}
        <div className="relative aspect-[3/4] bg-black">
          <Webcam
            ref={webcamRef}
            className="w-full h-full object-cover"
            mirrored
            screenshotFormat="image/jpeg"
            videoConstraints={{ facingMode: "user" }}
          />
          
          {/* Overlay */}
          <div className="absolute inset-0 pointer-events-none">
            {isDetecting && (
              <div className="absolute inset-0 border-4 border-blue-500/30 animate-pulse rounded-lg m-4"></div>
            )}
            
            {/* Error Toast */}
            {error && (
              <div className="absolute top-4 left-4 right-4 bg-red-500/90 text-white p-3 rounded-lg flex items-center gap-2 text-sm animate-in fade-in slide-in-from-top-4">
                <AlertTriangle className="w-4 h-4" />
                {error}
              </div>
            )}
            
            {/* Prediction Result */}
            {prediction && (
               <div className="absolute bottom-8 left-4 right-4">
                 <div className="bg-gray-900/90 backdrop-blur-md p-4 rounded-xl border border-white/10 shadow-2xl transform transition-all duration-300">
                   <div className="flex items-center gap-3 mb-2">
                     <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                       <CheckCircle className="w-5 h-5 text-green-400" />
                     </div>
                     <div>
                       <p className="text-xs text-gray-400 uppercase tracking-wider font-semibold">Detected Sign</p>
                       <h2 className="text-2xl font-bold text-white">{prediction}</h2>
                     </div>
                   </div>
                   <div className="w-full bg-gray-700 h-1.5 rounded-full overflow-hidden">
                     <div 
                       className="h-full bg-gradient-to-r from-blue-400 to-green-400 transition-all duration-300"
                       style={{ width: `${confidence * 100}%` }}
                     />
                   </div>
                   <p className="text-right text-xs text-gray-400 mt-1">{Math.round(confidence * 100)}% confidence</p>
                 </div>
               </div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="p-6 bg-gray-800">
          <p className="text-center text-gray-400 mb-6 text-sm">
            {isDetecting 
              ? "Camera is active. Show gestures to detect." 
              : "Press play to start real-time detection."}
          </p>
          
          <div className="flex justify-center">
            <button
              onClick={toggleDetection}
              disabled={isModelLoading}
              className={`
                w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg hover:scale-105 active:scale-95
                ${isDetecting 
                  ? 'bg-red-500 hover:bg-red-600 shadow-red-500/30' 
                  : 'bg-blue-600 hover:bg-blue-700 shadow-blue-600/30'
                }
                ${isModelLoading ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              {isDetecting ? (
                <Square className="w-8 h-8 fill-current" />
              ) : (
                <Play className="w-8 h-8 fill-current ml-1" />
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App;

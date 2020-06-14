const IMAGE_SIZE =100;
const modelUrl = "https://storage.googleapis.com/tfjsaldo/keras_model/model.json"

async function showSummary()   {
  try {
    window.alert("Please open your console")
    const model = await tf.loadLayersModel(modelUrl)
    console.log(model.summary())
    
  } catch (error) {
    console.error(error)
  }
}

async function readURL(input) {
  if (input.files && input.files[0]) {
      console.log(input.files)
      var reader = new FileReader();
      reader.onload = function (e) {
        $('#uploadedImg').attr('src', e.target.result);
      }
      reader.readAsDataURL(input.files[0]);
  }
}

$(document).ready(function(){
  $("#fileid").on('change',function(){
      readURL(this)
      
  });
});

function resizeImage(imgElement){
  const canvas = document.createElement('canvas');
  canvas.height=IMAGE_SIZE;
  canvas.width=IMAGE_SIZE;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height); // clear canvas
  ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

  const img = document.createElement('img');
  img.src = canvas.toDataURL("image/jpeg");;
  img.width = IMAGE_SIZE;
  img.height = IMAGE_SIZE;

  return img;
}

async function predict() {
  const img = document.getElementById("uploadedImg")
  const rezImg = resizeImage(img)
  console.log('predict img')

  const model = await tf.loadLayersModel(modelUrl);
  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    // const img = tf.browser.fromPixels(imgElement, 1);

    // Normalize the image from [0, 225] to [INPUT_MIN, INPUT_MAX]
    const normalizationConstant = 1.0 / 255.0;
    // const normalized = img.toFloat().mul(normalizationConstant);

    let tensor = tf.browser.fromPixels(rezImg, 3)
      .resizeBilinear([64, 64], false)
      .expandDims(0)
      .toFloat()
      .mul(normalizationConstant)

    startTime2 = performance.now();

    // Make a prediction through model.
    return model.predict(tensor);
  });

  // Convert logits to probabilities and class names.
  const classes = await logits.data();

  console.log('Predictions: ', classes);
  const labelName = classes[0] < 0.5? 'MASK' : 'NO_MASK';
  const probability = classes[0];

  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  console.log(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
      
 
  // Return the best probability label
  console.log(`${labelName} (${Math.floor(probability * 100)}%)`);
  
  document.getElementById("result").innerHTML = "";
  const res = `${labelName} (${Math.floor(probability * 100)}%)`
  var newElement = document.createElement("p");
  var node = document.createTextNode(res);
  newElement.appendChild(node);
  var element = document.getElementById("result");
  element.appendChild(newElement);
  
  return `${labelName} (${Math.floor(probability * 100)}%)`;
}
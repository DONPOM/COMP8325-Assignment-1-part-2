
// Initializes the class 
Shadowbox.init({
    // let's skip the automatic setup because we don't have any
    // properly configured link elements on the page
    skipSetup: true
   // overlayColor: "CCCCCC"

 
});


window.onload = function() {
// If the fromAd var is present the shadow box will open displaying the video, if not nothing happens
var viewAd = getQueryVariable('fromAd');
var getVideo;

 if(viewAd) {
//I use the name of the patient as the queryParameter
    if(viewAd == 'Servants') {
         initiateVideo("U4yZAcnFhyg");
    }
				                     
 };


//adds id to videoplayer and opens shadowbox
function initiateVideo(_videoId) {
				var s = "https://www.youtube.com/embed/";
				    s = s + _videoId;
				    s = s + "?autoplay=1&rel=0&enablejsapi=1&playerapiid=ytplayer";
				    
				    getVideo = s;    
				     openShadowbox();
}
 
// open a welcome message as soon as the window loads
function openShadowbox() {
   Shadowbox.open({
        content: getVideo,
        player:     "iframe",
        title:      "",
        height:     315,
        width:      560,
        resizeDuration: 0.75,
        handleUnsupported:  "remove",
        overlayColor: ""
    });
  }
};

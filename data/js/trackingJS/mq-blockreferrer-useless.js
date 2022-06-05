
var blockedReferrer = "";
var dom=location.hash
if (dom!=''){
  dom=dom.substr(1)
  document.domain=dom
}

function adsPageOnL(){
  var adFr=window.frameElement
  if (adFr && parent.adsCkCol){
      if (adFr.textAd!=1){
        var collapse=parent.adsCkCol(adFr,document)
        if (!collapse&&adFr.divName){
          var parDiv=parent.document.getElementById(adFr.divName)
          if (parDiv&&(parDiv.dynSz==1)&&parent.adsRMIFOnL){
          parent.adsRMIFOnL(window,document)
          }
        }
      }
      else
        parent.adsDoOnL(adFr,document)
  }
}


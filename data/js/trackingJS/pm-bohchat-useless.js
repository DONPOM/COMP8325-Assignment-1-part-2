
  var _bcvma = _bcvma || [];
  _bcvma.push(["setAccountID", "7371496673391179132"]);
  _bcvma.push(["setParameter", "WebsiteDefID", "4010568708627167909"]);
  _bcvma.push(["setParameter", "InvitationDefID", "807767743765768988"]);
  _bcvma.push(["setParameter", "VisitName", ""]);
  _bcvma.push(["setParameter", "VisitPhone", ""]);
  _bcvma.push(["setParameter", "VisitEmail", ""]);
  _bcvma.push(["setParameter", "VisitRef", ""]);
  _bcvma.push(["setParameter", "VisitInfo", ""]);
  _bcvma.push(["setParameter", "CustomUrl", ""]);
  _bcvma.push(["setParameter", "WindowParameters", ""]);
  _bcvma.push(["addFloat", {type: "chat", id: "7298435593501349359"}]);
  _bcvma.push(["pageViewed"]);
  var bcLoad = function(){
    if(window.bcLoaded) return; window.bcLoaded = true;
    var vms = document.createElement("script"); vms.type = "text/javascript"; vms.async = true;
    vms.src = ('https:'==document.location.protocol?'https://':'http://') + "vmss.boldchat.com/aid/7371496673391179132/bc.vms4/vms.js";
    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(vms, s);
  };
  if(window.pageViewer && pageViewer.load) pageViewer.load();
  else if(window.addEventListener) window.addEventListener('load', bcLoad, false);
  else window.attachEvent('onload', bcLoad);

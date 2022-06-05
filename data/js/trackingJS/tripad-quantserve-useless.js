
	function channValidator(chann){
	return (typeof(chann) == 'string' && chann != '');
	}
	function lycosQuantcast(){
		var lb = "";
		if(typeof(cm_host) !== 'undefined' && channValidator(cm_host)){
			lb += cm_host.split('.')[0] + '.';
		}
		if(typeof(cm_taxid) !==	'undefined' && channValidator(cm_taxid)){
			lb += cm_taxid;
			lb = lb.replace('/','');
		} else {
			lb = lb.replace('.','');
		}
		return lb;
	}
	var _qevents = _qevents || [];
	(function() {
		var elem = document.createElement('script');
		elem.src = (document.location.protocol == "https:" ? "https://secure" :"http://edge") + ".quantserve.com/quant.js";
		elem.async = true;
		elem.type = "text/javascript";
		var scpt = document.getElementsByTagName('script')[0];
		scpt.parentNode.insertBefore(elem, scpt);
	})();
	
	_qevents.push({
		qacct:"p-6eQegedn62bSo",
		labels:lycosQuantcast()
	});

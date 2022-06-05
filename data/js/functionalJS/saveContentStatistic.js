	function saveContentStatistic(cid) {
		var hostName=document.domain;
		jQuery.getJSON('remoteComponents/saveContentStatistic.cfc?method=saveContentView', {contentID:cid,website:hostName,logPeopleID:0}, function data() {
			
		})
	}
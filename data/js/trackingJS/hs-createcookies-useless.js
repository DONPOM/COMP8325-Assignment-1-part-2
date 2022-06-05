

$( function() {
  $('#text-site').click( function() {
      createCookie('version', 'text', 1);
      window.location.reload();
      return false;
  });

  var pathname = window.location.pathname;
  if(pathname.indexOf("/sports/") ==  -1) {
	$('.internal-page #rightcol').css('display','none');
	$('.internal-page #mainbody').css('width','970px');
  }
});


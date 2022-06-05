
function getParameterByName(name)
{
  name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
  var regexS = "[\\?&]" + name + "=([^&#]*)";
  var regex = new RegExp(regexS);
  var results = regex.exec(window.location.search);
  if(results == null)
    return "";
  else
    return decodeURIComponent(results[1].replace(/\+/g, " "));
}

ados_results = null;
ados.isAsync = true;
ados_add_placement(5146, 24950, "main", 5);
ados_add_placement(5146, 24950, "sponsorship", 8);
ados_setWriteResults(true);
var keywords = getParameterByName("sr");
if(keywords != "")
  ados_setKeywords(keywords);
ados_load();

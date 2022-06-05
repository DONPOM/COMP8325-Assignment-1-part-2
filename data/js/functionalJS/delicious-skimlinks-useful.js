function initSkimLinks(){
    $('.links .link-list a.title').click(function(event){
    if($(this).attr('skimlinks')!="1"){
        var currUrl = encodeURI($(this).attr('href'));
        var ref = encodeURI(document.location.href);
        
        var url = "http://go.delicious.com/?id=63408X1518852&xs=1&url="+ currUrl +"&sref="+ ref
        $(this).attr('href', url);
        $(this).attr('skimlinks',1)
        
        //event.preventDefault();
         //console.log($(this).attr('href'));
        //return false;
        }
    }
    );
  }

/*
if(window.applicationCache){  
  var appCache = window.applicationCache;
  appCache.update(); // Attempt to update the user's cache.
  if (appCache.status == window.applicationCache.UPDATEREADY) {
    appCache.swapCache();  // The fetch was successful, swap in the new cache.
  }
}
*/
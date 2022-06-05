
  var timeout = 0;
  var addFooter = function() {
    var image;
    if($('iframe').length > 0) {
      image = $('iframe').contents().find("img").first()[0];
    }
    else {
      image = $('#main img').first()[0];
    }
    
    if (image == undefined) {
      timeout++;
      if(timeout < 200) {
        setTimeout(addFooter, 10);
      }
    }
    else {
      var url = image.src;
      if (url) {
        $('#redditthis').append("<a style=\"font: small verdana,arial,sans-serif;color: #369;text-decoration: none;\" target=\"top\" href=\"http://ads.reddit.com/submit?url=" + encodeURIComponent(url) + "\">discuss this ad on reddit</a>");
      }
    }
  };
  $(window).load(addFooter);

         P.when("A", "ready").execute(function(A) {
             var $ = A.$;
             var $yt_title_block_left = $(".yt-title-block-left");
             var $yt_title_block_right = $(".yt-title-block-right");
             var do_not_capitalize =  {'a':1,'an':1,'the':1,'and':1,'but':1,'or':1,'for':1,'nor':1,'on':1,'at':1,'to':1,'from':1,'by':1};
             $yt_title_block_left.each(function() {
                 var title = $(this).text();
                 title = title.replace(/\w+/g, function(match) {
                     if(do_not_capitalize[match]) { return match; }
                     return match.substr(0,1).toUpperCase()+match.substr(1);
                 });
                 $(this).find("span").text(title);
             });
             $yt_title_block_right.find("a").text('See more');
             $yt_title_block_left.fadeIn();
             $yt_title_block_right.fadeIn();
         });
     
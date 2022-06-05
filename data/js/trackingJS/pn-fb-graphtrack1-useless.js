
    // user variables
    var myThumbUrl = '//graph.facebook.com/100001546764538/picture?type=square',
        myName = '&quot;Shohae Yu&quot;',
        myUID = '422403';

    var noBind = false;
    $(document).ready(function () {
                $('#startPopup').click();
                return false;
    });
    var noScroll = true;    // linkUrl is optional for most shares (besides Twitter)
function sharePage(url,type)
{
var shareHeight, shareWidth, shareX, shareY, shareCount;
switch(type)
{
case 'facebook':
shareWidth = 600;
shareHeight = 380;
break;
case 'twitter':
shareWidth = 800;
shareHeight = 320;
break;
case 'tumblr':
shareWidth = 450;
shareHeight = 435;
break;
case 'pin':
shareWidth = 600;
shareHeight = 300;
break;
case 'google':
shareWidth = 500;
shareHeight = 320;
break;
default:
return;
}

shareX = ((window.screen.width-shareWidth)/2);
shareY = ((window.screen.height - shareHeight)/2);
newwindow=window.open(url,'shareWindow','height='+shareHeight+',width='+shareWidth+',left='+shareX+',top='+shareY);
if (window.focus) {newwindow.focus()}
}

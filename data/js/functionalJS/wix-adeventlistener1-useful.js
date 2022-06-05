
                items_with_sub_menu = document.getElementsByClassName("sub_menu");
                if (isMobile)
                {
                    navigation_bar.setAttribute('mb','true');
                    for (var i = 0; i < items_with_sub_menu.length; i++)
                    {
                        var item = items_with_sub_menu[i].parentNode.parentNode;
                        item.setAttribute('mb','true');
                        item.children[0].addEventListener('click',mobileMenuClick,false);
                    }
                } else
                {
                    submenu = new Submenu(document.getElementsByClassName("menu_item"));
                }
                var arrow = document.createElement('div'); arrow.className = 'arrow'; arrow.appendChild(document.createElement('div'));
                for (var i = 0; i < items_with_sub_menu.length; i++)
                {
                    var item = items_with_sub_menu[i].parentNode.parentNode;
                    item.insertBefore(arrow.cloneNode(true),item.children[0]);
                }
                arrow = null;
            
AdobeGlobalNav._jsonpTemplateDataCallback({
    profile: {
        uploadAvatarURL: {
            defaultLink: "https://accounts.adobe.com/profile"
        },
        uploadAvatarIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><path d="M32 12.09l-.102 12.494c0 2.743-2.336 3.657-3.556 3.76H3.758c-1.22 0-2.133-.407-2.844-1.118C0 26.21 0 24.99 0 24.586v-12.19c0-2.744 2.438-3.86 3.962-3.963h2.64c.51-4.368 3.05-4.775 3.353-4.775h7.924c2.64 0 3.86 2.337 4.266 4.775h6.095c2.744 0 3.658 2.337 3.76 3.657 0-.1 0 0 0 0zm-1.32 12.596V12.19c0-.1-.204-2.437-2.44-2.437h-6.704c-.305 0-.61-.305-.61-.61-.102-1.016-.61-4.165-3.048-4.165h-7.72c-.103 0-2.033.305-2.236 4.165 0 .305-.305.61-.61.61h-3.25c-.103 0-2.642.102-2.642 2.64v12.293c0 .203 0 1.117.61 1.727.406.508 1.016.71 1.83.71h24.38c.102 0 2.438-.304 2.438-2.437zm-9.448-6.604c0 3.962-3.15 7.11-7.11 7.11s-7.112-3.148-7.112-7.11 3.15-7.11 7.11-7.11c3.86 0 7.112 3.148 7.112 7.11zm-1.32.102c0-3.15-2.54-5.79-5.79-5.79-3.15 0-5.79 2.54-5.79 5.79 0 3.15 2.54 5.79 5.79 5.79 3.148 0 5.79-2.64 5.79-5.79zm6.704-4.063c0 1.017-.813 1.93-1.93 1.93-1.016 0-1.93-.812-1.93-1.93 0-1.015.813-1.93 1.93-1.93s1.93.814 1.93 1.93zm-1.22 0c0-.405-.304-.71-.71-.71-.406 0-.71.305-.71.71 0 .407.304.712.71.712.407 0 .71-.305.71-.71z" fill="#fff"/></svg>',
        ccIcon: {
            hideForCountries: ["CN", "RS"],
            icon: '<svg xmlns="http://www.w3.org/2000/svg" width="64.6" height="47" viewBox="22.7 16.5 64.6 47" enable-background="new 22.7 16.5 64.6 47"><path fill="#ccc" d="M85.1 34.3c-.2-.8-.5-1.7-.9-2.5.1.3.2.5.3.8-.2-.5-.5-1.1-.7-1.6.1.3.2.5.3.8-2.5-6-7.6-10.6-13.7-12.7-6.3-2.1-13.4-1.2-19.1 2.4-2.2 1.4-4.1 3.1-5.8 5.1-5.9-1-12.1 1-16.3 5.3-4.1 4.2-5.9 10.2-4.9 16C25.8 56.5 33.7 63 42.4 63h21c6.5 0 12.8-2.8 17-7.7 4.5-5.1 6.4-12 5.2-18.6-.1-.8-.3-1.7-.5-2.4 0-.2.5 2 0 0zm-31.4.3l6.7 7.5c1.5 1.6-1.1 3.8-2.5 2.3l-6.8-7.5C47 32.3 39.8 31.7 35 35.6s-5.7 11.1-2 16c2.2 2.9 5.8 4.7 9.4 4.7h4.9c1.3 1.3 2.7 2.4 4.2 3.3h-9.1c-6.2 0-12-4-14.1-9.8-2.1-5.7-.5-12.2 3.9-16.3 6.1-5.5 16-5 21.5 1.1zm2 23.5c-2.3-1-4.4-2.5-6.2-4.3l-8-8.1c-1.5-1.5.9-3.9 2.4-2.3l8 8.1c4.2 4.3 10.6 5.9 16.3 4.1 5.8-1.9 10-7 10.8-12.9.8-6-2-12-7-15.4-5.2-3.5-12.3-3.5-17.6 0-.8.5-1.5 1.1-2.3 1.8-1-.6-2-1.1-3.2-1.6 4.4-4.8 11.1-7.1 17.5-6 6 1 11.2 4.9 13.9 10.3 2.7 5.3 2.7 11.9 0 17.3s-8 9.3-13.9 10.3c-3.5.6-7.2.2-10.7-1.3-2.3-1 2.4 1 0 0z"/></svg>'
        },
        actions: [{
            id: "manageAccount",
            title: "Manage Account",
            url: {
                defaultLink: "https://accounts.adobe.com"
            }
        }, {
            id: "manageTeam",
            entitlements: ["creativeCloudTeamAdmin"],
            title: "Manage Team",
            url: {
                defaultLink: "https://creative.adobe.com/team/admin",
                languageCodeType: "defaultSet",
                defaultLanguageLink: "https://creative.adobe.com/team/admin?locale=@@LANGUAGE@@"
            }
        }
        ],
        groups: [{
            hideForCountries: ["CN", "RS"],
            entitlements: ["creativeCloudUser"],
            title: "Apps",
            items: [{
                id: "desktop",
                title: "Desktop",
                url: {
                    defaultLink: "https://www.adobe.com/creativecloud/catalog/desktop.html",
                    localeCodeType: "defaultSet",
                    defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/catalog/desktop.html",
                    excludeLocales: ["sr_RS", "zh_CN"]
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 19" enable-background="new 0 0 24 19"><g stroke="#231F20" stroke-miterlimit="10" fill="none"><path d="M23.5 14.5c0 .55-.45 1-1 1h-21c-.55 0-1-.45-1-1v-13c0-.55.45-1 1-1h21c.55 0 1 .45 1 1v13zM2.5 2.5h19v11h-19z"/><path stroke-linecap="round" d="M7 18.5h10"/><path d="M10.5 15.5v3M13.5 15.5v3"/></g></svg>'
                }
            }, {
                id: "addons",
                title: "Add-ons",
                url: {
                    defaultLink: "https://creative.adobe.com/addons",
                    languageCodeType: "defaultSet",
                    defaultLanguageLink: "https://creative.adobe.com/addons?locale=@@LANGUAGE@@"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 19" enable-background="new 0 0 24 19"><path fill-rule="evenodd" clip-rule="evenodd" d="M9.418 15.847h-7.24V3.097H7.89l8.524 6.446-2.306 1.89 1.657 1.277 3.82-3.286L8.434.97H-.01v17h12.253l-2.825-2.123zm5.32-12.75h7.135v12.75h-5.535l-8.63-6.51 2.236-1.943-1.53-1.16-3.84 3.286 11.206 8.452h8.21v-17H11.953l2.785 2.125z"/></svg>'
                }
            }, {
                id: "mobile",
                title: "Mobile",
                url: {
                    defaultLink: "https://www.adobe.com/creativecloud/catalog/mobile.html",
                    localeCodeType: "defaultSet",
                    defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/catalog/mobile.html",
                    excludeLocales: ["sr_RS", "zh_CN"]
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 24" enable-background="new 0 0 18 24"><path stroke="#231F20" stroke-linecap="round" stroke-miterlimit="10" d="M17.5 21.5c0 1.104-.895 2-2 2h-13c-1.104 0-2-.896-2-2v-19c0-1.105.896-2 2-2h13c1.105 0 2 .895 2 2v19z" fill="none"/><path fill="#231F20" d="M9 20.5c-.553 0-1 .447-1 1 0 .552.447 1 1 1 .552 0 1-.448 1-1 0-.553-.448-1-1-1z"/><path stroke="#231F20" stroke-linecap="round" stroke-linejoin="round" stroke-miterlimit="10" fill="none" d="M2.5 3.5h13v16h-13z"/><path fill="#231F20" d="M9 1.5c-.276 0-.5.224-.5.5s.224.5.5.5.5-.224.5-.5-.224-.5-.5-.5z"/></svg>'
                }
            }
            ]
        }, {
            hideForCountries: ["CN", "RS"],
            entitlements: ["creativeCloudUser"],
            title: "Assets",
            items: [{
                id: "files",
                title: "Files",
                url: {
                    defaultLink: "https://assets.adobe.com/files",
                    languageCodeType: "defaultSet",
                    defaultLanguageLink: "https://assets.adobe.com/files?locale=@@LANGUAGE@@"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" enable-background="new 0 0 24 24"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;fill:#666767;}</style><path class="st0" d="M15.7 12.8L11.9 9l-3.8 3.8.8.7 2.4-2.5v9.5h1.1V11l2.4 2.5.9-.7zm7.1-1.9c0 3.5-2.8 5.2-4.8 5.2h-2.3V15H18c1.3 0 3.7-1.1 3.7-4.1 0-2.8-2.7-4.1-5.2-3.5-.3-2.4-2.3-3.7-5-3.7-2.8 0-5 2-5.4 4.8l-.2.8C4.4 9.1 2 9.9 2 12.1 2 13.9 3.3 15 5.1 15h2.7v1.1H5.2c-2.1 0-4.2-1.3-4.2-4 0-1.7 1.3-3.9 4.1-3.9.4-3.2 3-5.7 6.4-5.7 2.5 0 5.2 1.3 5.8 3.8 2.7-.3 5.5 1.5 5.5 4.6z"/></svg>'
                }
            }, {
                id: "photos",
                title: "Photos",
                url: {
                    defaultLink: "https://lightroom.adobe.com"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" enable-background="new 0 0 24 24"><style type="text/css">.st0{fill:#676767;}</style><path class="st0" d="M20.6 3.6v16.9H3.4V3.6h17.2m1-1.1H2.4v19h19.3v-19h-.1zM7.5 16.1c-.1 0-.1 0-.1-.1V7.6c0-.1 0-.1.1-.1h1.2c.1 0 .1 0 .1.1v7.3h3.1c.1 0 .1 0 .1.1l-.3 1c0 .1-.1.1-.1.1H7.5zm5.9-5.1v-.8c0-.1 0-.1.1-.1.5-.2 1.8-.5 3.1-.5.1 0 .1 0 .1.1v1.1c0 .1 0 .1-.1.1-.5 0-1.6 0-1.8.1v5c0 .1 0 .1-.1.1h-1.1c-.1 0-.1 0-.1-.1v-5z"/></svg>'
                }
            }, {
                id: "fonts",
                title: "Fonts",
                url: {
                    defaultLink: "https://typekit.com/fonts?ref=globalnav",
                    languageCodeLinks: {
                        en: "https://typekit.com/fonts?locale=en-US&ref=globalnav"
                    },
                    localeCodeType: "typekit",
                    defaultLocaleLink: "https://typekit.com/fonts?locale=@@LOCALE@@&ref=globalnav"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="100 0 400 400" enable-background="new 100 0 400 400"><path fill="#fff" d="M409.52 30l47.6 48.96v81.6h-81.6V370H274.88l-43.52-42.16V160.56h-40.8l-38.08-48.96h95.2v209.44h82.96V111.6h78.88z"/></svg>'
                }
            }, {
                id: "colors",
                title: "Colors",
                url: {
                    defaultLink: "https://kuler.adobe.com/",
                    languageCodeType: "defaultSet",
                    defaultLanguageLink: "https://kuler.adobe.com/@@LANGUAGE@@/"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" enable-background="new 0 0 24 24"><style type="text/css">.st0{fill:#676767;}</style><path class="st0" d="M18.2 18.4l-1.5-1.5c1.3-1.5 2-3.4 2-5.5s-.8-4-2-5.5l1.4-1.4c.8.5 2 .4 2.7-.3.8-.8.8-2.3 0-3.1-.8-.8-2.4-.8-3.2.1-.7.7-.7 1.7-.2 2.5L16 5.1c-1.4-1.2-3.3-2-5.3-2-4.5 0-8.2 3.7-8.2 8.3 0 4.6 3.7 8.3 8.2 8.3 2 0 3.9-.7 5.3-2l1.5 1.5c-.4.8-.3 1.9.4 2.6.8.8 2.2.8 3 0 .8-.8.8-2.2 0-3-.8-.7-1.9-.9-2.7-.4zm-1.1-6.9c0 1.6-.6 3.1-1.6 4.3l-3.2-3.2c.2-.3.3-.7.3-1 0-.4-.1-.8-.3-1.1l3.2-3.2c1 1 1.6 2.5 1.6 4.2zM10.6 18c-3.6 0-6.5-2.9-6.5-6.6S7 4.9 10.6 4.9c1.5 0 2.9.5 4.1 1.4l-3.2 3.2c-.3-.2-.7-.3-1-.3-1.2 0-2.2 1-2.2 2.2 0 1.2 1 2.2 2.2 2.2.4 0 .8-.1 1.1-.3l3.2 3.2c-1.2 1-2.6 1.5-4.2 1.5z"/></svg>'
                }
            }, {
                id: "libraries",
                title: "Libraries",
                url: {
                    defaultLink: "https://assets.adobe.com/assets/libraries",
                    languageCodeType: "defaultSet",
                    defaultLanguageLink: "https://assets.adobe.com/assets/libraries?locale=@@LANGUAGE@@"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" enable-background="new 0 0 24 24"><style type="text/css">.st0{fill-rule:evenodd;clip-rule:evenodd;fill:#666767;}</style><path class="st0" d="M2.9 2.5v15h3v3.1h15V5.5h-3v-3h-15zm1.1 1h12.9v2h-11v10.9H4V3.5zm13.9 3h1.9v12.9h-13V6.5h7V14l1.9-1.6 2.1 1.7V6.5z"/></svg>'
                }
            }
            ]
        }, {
            hideForCountries: ["CN", "RS"],
            entitlements: ["creativeCloudUser"],
            title: "Community",
            items: [{
                id: "creativeNetwork",
                title: "Creative Network",
                url: {
                    defaultLink: "https://www.behance.net?sso_inbound=1"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 -100 600 600" enable-background="new 0 -100 600 600"><path d="M174.905 11.544c17.888 0 33.81 1.518 48.556 4.723 14.406 3.1 26.8 8.1 37.572 15.26 10.13 7.073 18.188 16.434 24.193 28.36 5.578 11.582 8.292 26.222 8.292 43.554 0 18.616-4.188 34.196-12.694 46.698-8.335 12.46-21.2 22.697-37.892 30.647 22.933 6.54 39.837 18.123 51.12 34.493 11.2 16.585 16.713 36.438 16.713 59.627 0 18.85-3.525 34.964-10.813 48.557-7.31 13.763-16.926 25.047-29.322 33.596-12.417 8.826-26.394 15.152-42.38 19.276-15.56 4.19-31.694 6.306-48.407 6.306H0V11.544h174.905zm-10.45 149.9c14.574 0 26.393-3.42 35.796-10.344 9.597-6.882 14.085-18.038 14.085-33.917 0-8.484-1.454-15.558-4.467-21.093-3.27-5.75-7.438-9.917-12.63-12.952-5.3-3.333-11.328-5.386-18.102-6.56-6.71-1.284-13.72-1.796-20.987-1.796H81.937v86.662h82.516zm4.615 157.914c7.843 0 15.537-.77 22.718-2.224 7.33-1.71 13.72-4.23 19.213-7.907 5.708-3.633 10.11-8.57 13.4-14.875 3.314-6.304 4.938-14.21 4.938-24.063 0-19.064-5.493-32.593-16.2-40.99-10.707-8.037-25.004-12.12-42.614-12.12H81.938v102.18h87.132zM422.217 318.14c11.006 10.835 27.163 16.114 47.828 16.114 14.81 0 27.934-3.654 38.556-11.112 10.772-7.524 17.247-15.623 19.662-23.83h65.076c-10.558 32.25-26.394 55.352-47.893 69.31-21.565 13.976-47.467 20.836-77.9 20.836-21.2 0-39.965-3.44-57.19-10.152-17.097-6.796-31.416-16.434-43.49-28.81-11.754-12.63-20.646-27.353-27.228-44.815-6.54-17.33-9.617-36.458-9.617-57.295 0-19.983 3.29-38.833 9.896-56.08 6.624-17.48 15.943-32.42 28.018-45.05 12.31-12.738 26.715-22.697 43.49-30.07 16.863-7.18 35.563-10.942 56.08-10.942 22.803 0 42.68 4.488 59.712 13.357 16.99 8.826 31.03 20.794 41.866 35.626 11.007 14.768 18.85 31.865 23.616 50.907 4.915 19.148 6.604 39.088 5.15 59.84H404.05c.043 21.22 7.096 41.438 18.167 52.167zm84.93-141.095c-8.784-9.724-23.702-14.94-41.782-14.94-11.904 0-21.5 1.882-29.514 5.92-7.544 4.02-13.763 9.02-18.55 14.876-4.787 5.985-7.865 12.268-10.066 18.893-1.88 6.604-2.99 12.503-3.163 17.76h119.94c-1.668-18.914-8.06-32.763-16.863-42.508zM388.854 36.377h150.414V73.05H388.854z"/></svg>'
                }
            }, {
                id: "portfolioWebsite",
                itemVariants: [],
                title: "Portfolio Website",
                url: {
                    defaultLink: "https://www.behance.net/prosite?sso_inbound=1"
                },
                hoverIcon: {
                    icon: '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="100 0 400 400" enable-background="new 100 0 400 400"><path fill="#969595" d="M368.264 80.478c-16.658-11.4-36.977-13.798-51.1-13.798H203.068v266.64h26.49V219.352h81.884c26.65 0 47.726-6.615 62.638-19.656 14.943-13.072 22.853-32.423 22.853-55.97 0-28.932-9.656-50.218-28.67-63.248zm-54.32 114.506H229.56V91.047h86.54c50.43 0 54.35 40.048 54.35 52.32-.01 38.06-29.192 51.617-56.505 51.617z"/></svg>'
                }
            }
            ]
        }
        ],
        signOut: {
            title: "Sign Out"
        },
        honorific: ""
    },
    sitemap: {
        sectionLists: {
            sectionListsVariants: [],
            items: [{
                sections: [{
                    id: "products",
                    title: "Products"
                }, {
                    id: "how-to-buy",
                    title: "How to Buy"
                }
                ]
            }, {
                sections: [{
                    id: "learn-and-support",
                    title: "Learn & Support"
                }, {
                    id: "about-adobe",
                    title: "About Adobe"
                }
                ]
            }
            ]
        },
        sections: [{
            id: "products",
            groups: [{
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "leading-item",
                    hideForCountries: ["CN", "RS"]
                }, {
                    variant: "top-spacing-3"
                }, {
                    hideForCountries: ["CN", "RS"],
                    variant: "bottom-spacing-1"
                }, {
                    showForCountries: ["CN", "RS"],
                    variant: "bottom-spacing-2"
                }
                ],
                items: [{
                    hideForCountries: ["CN", "RS"],
                    iconAndText: {
                        id: "creativeCloud",
                        itemVariants: [{
                            variant: "leading-item"
                        }, {
                            variant: "creative-cloud"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" width="64.6" height="47" viewBox="22.7 16.5 64.6 47" enable-background="new 22.7 16.5 64.6 47"><path fill="#ccc" d="M85.1 34.3c-.2-.8-.5-1.7-.9-2.5.1.3.2.5.3.8-.2-.5-.5-1.1-.7-1.6.1.3.2.5.3.8-2.5-6-7.6-10.6-13.7-12.7-6.3-2.1-13.4-1.2-19.1 2.4-2.2 1.4-4.1 3.1-5.8 5.1-5.9-1-12.1 1-16.3 5.3-4.1 4.2-5.9 10.2-4.9 16C25.8 56.5 33.7 63 42.4 63h21c6.5 0 12.8-2.8 17-7.7 4.5-5.1 6.4-12 5.2-18.6-.1-.8-.3-1.7-.5-2.4 0-.2.5 2 0 0zm-31.4.3l6.7 7.5c1.5 1.6-1.1 3.8-2.5 2.3l-6.8-7.5C47 32.3 39.8 31.7 35 35.6s-5.7 11.1-2 16c2.2 2.9 5.8 4.7 9.4 4.7h4.9c1.3 1.3 2.7 2.4 4.2 3.3h-9.1c-6.2 0-12-4-14.1-9.8-2.1-5.7-.5-12.2 3.9-16.3 6.1-5.5 16-5 21.5 1.1zm2 23.5c-2.3-1-4.4-2.5-6.2-4.3l-8-8.1c-1.5-1.5.9-3.9 2.4-2.3l8 8.1c4.2 4.3 10.6 5.9 16.3 4.1 5.8-1.9 10-7 10.8-12.9.8-6-2-12-7-15.4-5.2-3.5-12.3-3.5-17.6 0-.8.5-1.5 1.1-2.3 1.8-1-.6-2-1.1-3.2-1.6 4.4-4.8 11.1-7.1 17.5-6 6 1 11.2 4.9 13.9 10.3 2.7 5.3 2.7 11.9 0 17.3s-8 9.3-13.9 10.3c-3.5.6-7.2.2-10.7-1.3-2.3-1 2.4 1 0 0z"/></svg>'
                        },
                        title: "Creative Cloud",
                        url: {
                            defaultLink: "https://www.adobe.com/creativecloud.html?promoid=KLXLR",
                            countryCodeLinks: {
                                HR: "https://www.adobe.com/hr/creativecloud.html"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud.html",
                            excludeLocales: ["sr_RS", "zh_CN"]
                        }
                    }
                }, {
                    iconAndText: {
                        id: "photoshop",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 234" enable-background="new 0 0 240 234"><radialGradient id="a" cx="-183.829" cy="512.474" r=".76" gradientTransform="matrix(220 0 0 -213 40525 109215)" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#34364E" stop-opacity=".98"/><stop offset="1" stop-color="#0C0824"/></radialGradient><path fill="url(#a)" d="M10 10h220v214H10z"/><path fill="#31C5F0" d="M10 10h220v214H10V10zM0 234h240V0H0v234zM161.8 96c-7.9 0-10.6 4-10.6 7.3 0 3.6 1.8 6.1 12.4 11.6 15.7 7.6 20.6 14.9 20.6 25.6 0 16-12.2 24.6-28.7 24.6-8.7 0-16.2-1.8-20.5-4.3-.7-.3-.8-.8-.8-1.6v-14.7c0-1 .5-1.3 1.2-.8 6.3 4.1 13.5 5.9 20.1 5.9 7.9 0 11.2-3.3 11.2-7.8 0-3.6-2.3-6.8-12.4-12-14.2-6.8-20.1-13.7-20.1-25.2 0-12.9 10.1-23.6 27.6-23.6 8.6 0 14.6 1.3 17.9 2.8.8.5 1 1.3 1 2v13.7c0 .8-.5 1.3-1.5 1-4.4-2.8-10.9-4.5-17.4-4.5zm-86 11.9c2.3.2 4.1.2 8.1.2 11.7 0 22.7-4.1 22.7-20 0-12.7-7.9-19.1-21.2-19.1-4 0-7.8.2-9.6.3v38.6zM58 54.8c0-.7 1.4-1.2 2.2-1.2 6.4-.3 15.9-.5 25.8-.5 27.7 0 38.5 15.2 38.5 34.6 0 25.4-18.4 36.3-41 36.3-3.8 0-5.1-.2-7.8-.2v38.4c0 .8-.3 1.2-1.2 1.2H59.2c-.8 0-1.2-.3-1.2-1.2V54.8z"/></svg>'
                        },
                        title: "Photoshop",
                        url: {
                            defaultLink: "https://www.adobe.com/products/photoshop.html?promoid=KLXLS",
                            countryCodeLinks: {
                                CN: "https://www.adobe.com/cn/products/photoshop.html",
                                HR: "https://www.adobe.com/hr/products/photoshop.html",
                                RS: "https://www.adobe.com/rs/products/photoshop.html"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/photoshop.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "illustrator",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 234" enable-background="new 0 0 240 234"><radialGradient id="a" cx="-183.829" cy="512.474" r=".76" gradientTransform="matrix(220 0 0 -213 40525 109215)" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#423325" stop-opacity=".98"/><stop offset="1" stop-color="#1C0A00"/></radialGradient><path fill="url(#a)" d="M10 10h220v214H10z"/><path fill="#FF7F18" d="M10 10h220v214H10V10zM0 234h240V0H0v234zM157.2 83.4c0-.8.3-1.2 1.2-1.2h15.7c.8 0 1.2.3 1.2 1.2v78.9c0 .8-.2 1.2-1.2 1.2h-15.5c-1 0-1.3-.5-1.3-1.3V83.4h-.1zm-1.1-22.7c0-6.4 4.5-10.2 10.2-10.2 6.1 0 10.2 4.1 10.2 10.2 0 6.6-4.3 10.2-10.4 10.2-5.8 0-10-3.6-10-10.2zm-45 55.6c-2.8-11.1-9.4-35.3-11.9-47H99c-2.1 11.7-7.4 31.5-11.6 47h23.7zm-27.8 16.2l-7.9 30c-.2.8-.5 1-1.5 1H59.2c-1 0-1.2-.3-1-1.5l28.4-99.4c.5-1.8.8-3.4 1-8.3 0-.7.3-1 .8-1h21c.7 0 1 .2 1.2 1l31.8 107.9c.2.8 0 1.3-.8 1.3H125c-.8 0-1.3-.2-1.5-.9l-8.2-30.1h-32z"/></svg>'
                        },
                        title: "Illustrator",
                        url: {
                            defaultLink: "https://www.adobe.com/products/illustrator.html?promoid=KLXLT",
                            countryCodeLinks: {
                                CN: "https://www.adobe.com/cn/products/illustrator.html",
                                HR: "https://www.adobe.com/hr/products/illustrator.html",
                                RS: "https://www.adobe.com/rs/products/illustrator.html"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/illustrator.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "inDesign",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 234" enable-background="new 0 0 240 234"><radialGradient id="a" cx="-183.829" cy="512.474" r=".76" gradientTransform="matrix(220 0 0 -213 40525 109215)" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#4B2B36" stop-opacity=".98"/><stop offset="1" stop-color="#250012"/></radialGradient><path fill="url(#a)" d="M10 10h220v214H10z"/><path fill="#FF408C" d="M10 10h220v214H10V10zM0 234h240V0H0v234zM151.8 97.2c-1.5-.7-3.5-1-5.9-1-12.9 0-21.9 9.9-21.9 26.4 0 18.8 9.2 26.4 20.7 26.4 2.5 0 5.1-.3 7.1-1.2V97.2zm-5.3-16.3c2.1 0 3.3 0 5.3.2V54.5c0-.7.5-1 1-1H169c.8 0 1 .3 1 .8v93.5c0 2.8 0 6.3.5 10.1 0 .7-.2.8-.9 1.2-8.6 4.1-17.7 5.9-26.2 5.9-22.1 0-37.9-13.7-37.9-41.6-.1-23.9 15.7-42.5 41-42.5zm-58.7 81.3c0 1-.2 1.3-1.3 1.3H71.1c-1 0-1.3-.5-1.3-1.3V54.8c0-1 .5-1.3 1.3-1.3h15.5c.8 0 1.2.3 1.2 1.2v107.5z"/></svg>'
                        },
                        title: "InDesign",
                        url: {
                            defaultLink: "https://www.adobe.com/products/indesign.html?promoid=KLXLU",
                            countryCodeLinks: {
                                CN: "https://www.adobe.com/cn/products/indesign.html",
                                HR: "https://www.adobe.com/hr/products/indesign.html",
                                RS: "https://www.adobe.com/rs/products/indesign.html"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/indesign.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "premiere",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 234" enable-background="new 0 0 240 234"><radialGradient id="a" cx="-183.829" cy="512.474" r=".76" gradientTransform="matrix(220 0 0 -213 40525 109215)" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#4B2D4F" stop-opacity=".98"/><stop offset="1" stop-color="#180024"/></radialGradient><path fill="url(#a)" d="M10 10h220v214H10z"/><path fill="#E579FF" d="M10 10h220v214H10V10zM0 234h240V0H0v234zM144 98.8c0-1.2 0-4.1-.5-9.7 0-.8.1-1 .8-1.3 6.1-2.5 20.4-6.9 36.6-6.9.8 0 1.2.2 1.2 1v14.5c0 .8-.3 1-1.1 1-6.3-.3-15.4.5-18.9 2v63c0 .8-.3 1.2-1.2 1.2h-15.7c-.8 0-1.2-.3-1.2-1.2V98.8zm-64 9.1c2.3.2 4.1.2 8.1.2 11.7 0 22.7-4.1 22.7-20 0-12.7-7.8-19.1-21.2-19.1-4 0-7.8.2-9.6.3v38.6zM62.3 54.8c0-.7.2-1.2 1-1.2 6.4-.3 15.8-.5 25.7-.5 27.7 0 39.8 15.2 39.8 34.7 0 25.4-18.4 36.3-41 36.3-3.8 0-5.1-.2-7.8-.2v38.4c0 .8-.3 1.2-1.2 1.2H63.4c-.8 0-1.2-.3-1.2-1.2V54.8h.1z"/></svg>'
                        },
                        title: "Premiere Pro",
                        url: {
                            defaultLink: "https://www.adobe.com/products/premiere.html?promoid=KLXLV",
                            countryCodeLinks: {
                                CN: "https://www.adobe.com/cn/products/premiere.html",
                                HR: "https://www.adobe.com/hr/products/premiere.html",
                                RS: "https://www.adobe.com/rs/products/premiere.html"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/premiere.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "afterEffects",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 234" enable-background="new 0 0 240 234"><radialGradient id="a" cx="-183.829" cy="512.474" r=".76" gradientTransform="matrix(220 0 0 -213 40525 109215)" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#442F55" stop-opacity=".98"/><stop offset="1" stop-color="#12002C"/></radialGradient><path fill="url(#a)" d="M10 10h220v214H10z"/><g fill="#D8A1FF"><path d="M10 10h220v214H10V10zM0 234h240V0H0v234zM169.8 112.8c5.1 0 6.9 0 7.4-.2 0-.7.2-1.2.2-1.7 0-5.4-2.6-15.4-13-15.4-9.6 0-13.7 8.4-14.7 17.3h20.1zm-20.3 13.7c.2 13.5 6.6 22.6 21.8 22.6 5.9 0 11-.8 16.3-3.1.7-.3 1.2-.2 1.2.7v12.5c0 1-.3 1.5-1 2-5.3 2.6-11.9 3.8-20.1 3.8-26.4 0-36.3-19.5-36.3-41.2 0-23.6 12.2-42.9 33.7-42.9 21.8 0 29.4 18.3 29.4 33.2 0 4.8-.3 8.7-.8 10.6-.2.8-.5 1.1-1.3 1.3-2 .3-7.9.7-16.7.7h-26.2v-.2zm-53.6-10.2c-2.8-11.1-9.6-35.3-12.1-47h-.2c-2.1 11.7-7.6 31.5-11.7 47h24zm-28.1 16.2l-7.9 29.9c-.2.8-.5 1.1-1.5 1.1H43.7c-1 0-1.2-.3-1-1.5l28.4-99.3c.5-1.8.8-3.2 1-8.2 0-.7.3-1 .8-1h21c.7 0 1 .2 1.2 1l31.8 107.7c.2.8 0 1.3-.8 1.3h-16.5c-.8 0-1.3-.3-1.5-1l-8.2-30H67.8z"/></g></svg>'
                        },
                        title: "After Effects",
                        url: {
                            defaultLink: "https://www.adobe.com/products/aftereffects.html?promoid=KLXLW",
                            countryCodeLinks: {
                                CN: "https://www.adobe.com/cn/products/aftereffects.html",
                                HR: "https://www.adobe.com/hr/products/aftereffects.html",
                                RS: "https://www.adobe.com/rs/products/aftereffects.html"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/aftereffects.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "lightroom",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 234" enable-background="new 0 0 240 234"><radialGradient id="a" cx="-183.829" cy="512.474" r=".76" gradientTransform="matrix(220 0 0 -213 40525 109215)" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#39464D" stop-opacity=".98"/><stop offset="1" stop-color="#05151F"/></radialGradient><path fill="url(#a)" d="M10 10h220v214H10z"/><path fill="#AAD1EB" d="M10 10h220v214H10V10zM0 234h240V0H0v234zM140.5 98.5c0-1.2 0-4.1-.5-9.7 0-.8.2-1 .8-1.3 6.1-2.5 20.6-6.9 36.8-6.9.8 0 1.2.2 1.2 1V96c0 .8-.3 1-1.2 1-6.3-.3-15.6.5-19.1 2v63.1c0 .8-.3 1.2-1.2 1.2h-15.7c-.8 0-1.2-.3-1.2-1.2V98.5h.1zm-74 64.8c-1.2 0-1.5-.5-1.5-1.5V54.3c0-.8.3-1.3 1.2-1.3H82c.8 0 1 .3 1 1.2v92.6h41.2c.8 0 1.1.3.9 1.2l-2.5 14.2c-.2.8-.7 1.2-1.5 1.2H66.5v-.1z"/></svg>'
                        },
                        title: "Lightroom",
                        url: {
                            defaultLink: "https://www.adobe.com/products/photoshop-lightroom.html?promoid=KLXLX",
                            countryCodeLinks: {
                                CN: "https://www.adobe.com/cn/products/photoshop-lightroom.html",
                                HR: "https://www.adobe.com/hr/products/photoshop-lightroom.html",
                                RS: "https://www.adobe.com/rs/products/photoshop-lightroom.html"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/photoshop-lightroom.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "seeAll",
                        itemVariants: [{
                            variant: "small-icon"
                        }, {
                            variant: "tighten-horizontal-spacing-3"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="48 49.6 100 100" enable-background="new 48 49.6 100 100"><circle cx="65.1" cy="99.6" r="10"/><circle cx="98" cy="99.6" r="10"/><circle cx="130.9" cy="99.6" r="10"/></svg>'
                        },
                        title: "See all",
                        url: {
                            defaultLink: "https://www.adobe.com/creativecloud/catalog/desktop.html?promoid=KOVFF",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/catalog/desktop.html"
                        }
                    }
                }
                ]
            }, {
                hideForCountries: ["CN", "RS"],
                groupVariants: [{
                    variant: "bottom-spacing-2"
                }, {
                    variant: "equiwidth"
                }
                ],
                items: [{
                    htmlItem: {
                        id: "plansForEveryone",
                        itemVariants: [{
                            variant: "small"
                        }
                        ],
                        html: "<div class='globalnav__sitemap__item__title'><div>See plans for <a class='globalnav__sitemap__item__link globalnav__sitemap__item__call-to-action-link' href='{{#chooseURLForLocale}}{{{url1}}}{{/chooseURLForLocale}}'>small and medium businesses</a>, and <a class='globalnav__sitemap__item__link globalnav__sitemap__item__call-to-action-link' href='{{#chooseURLForLocale}}{{{url2}}}{{/chooseURLForLocale}}'>more ></a></div></div>",
                        url1: {
                            defaultLink: "https://www.adobe.com/creativecloud/buy/business.html?promoid=KQQSE",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/buy/business.html",
                            excludeLocales: ["sr_RS", "zh_CN"]
                        },
                        url2: {
                            defaultLink: "https://www.adobe.com/creativecloud.html?promoid=KPFMB#buy",
                            countryCodeLinks: {
                                HR: "https://www.adobe.com/hr/creativecloud.html#buy"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud.html#buy",
                            excludeLocales: ["sr_RS", "zh_CN"]
                        }
                    }
                }
                ]
            }, {
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "leading-item"
                }, {
                    variant: "border"
                }, {
                    hideForCountries: ["CN", "RS"],
                    variant: "bottom-spacing-1"
                }, {
                    showForCountries: ["CN", "RS"],
                    variant: "bottom-spacing-2"
                }, {
                    hideForCountries: ["CN", "RS"],
                    variant: "top-spacing-2"
                }, {
                    showForCountries: ["CN", "RS"],
                    variant: "top-spacing-3"
                }, {
                    showForCountries: ["CN"],
                    variant: "tighten-horizontal-spacing-2"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "marketingCloud",
                        itemVariants: [{
                            variant: "leading-item"
                        }, {
                            showForCountries: ["CN"],
                            variant: "nudge-from-start-1"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 196 199.2" enable-background="new 0 0 196 199.2"><path d="M97.7 136.5c-13.5 0-26.1-7.1-33.2-18.6l3.3-2c6.4 10.4 17.7 16.8 29.9 16.8h.8c12.1-.3 23-6.7 29.2-17.1l1.7 1-1.7-1c3.4-5.6 5-12.1 4.9-18.7-.2-9.4-4.1-18-10.8-24.5-6.8-6.4-15.6-9.9-25-9.6l-.1-3.8c10.4-.2 20.2 3.6 27.7 10.7 7.5 7.2 11.8 16.8 12 27.2.2 7.3-1.7 14.5-5.4 20.7-6.9 11.6-19 18.6-32.5 19l-.8-.1zm-81.6 15C6 136.2.5 118.4 0 100c-.6-26.1 9-50.9 27-69.7C45 11.4 69.3.7 95.4 0c26.1-.6 50.8 8.9 69.7 27 18.9 18 29.6 42.3 30.2 68.4.4 18.4-4.3 36.5-13.7 52.2l-3.3-2c9-15.1 13.5-32.5 13.1-50.1-.6-25.1-10.9-48.4-29.1-65.7-18.1-17.3-41.9-26.5-67-25.9l-.1-2.3.1 2.3c-25.1.6-48.4 10.9-65.7 29C12.5 51 3.3 74.8 3.9 99.9c.4 17.7 5.8 34.8 15.5 49.5l-3.3 2.1zm81.6 16.1c-23.4 0-45.4-11.9-58.4-31.4-7.2-10.9-11.2-23.7-11.5-36.9-.9-38.5 29.7-70.6 68.2-71.5l.1 3.8c-36.4.9-65.3 31.2-64.5 67.6.3 11.7 3.6 22.9 9.6 32.8 12.3 20.1 34.5 32.2 58 31.7 22.9-.5 43.5-12.6 55.2-32.3l3.3 2c-12.4 20.8-34.2 33.6-58.4 34.2h-1.6M62.1 121.2c-6.3-9.5-8.5-20.9-6.2-32 2.3-11.2 8.8-20.8 18.3-27 6.7-4.4 14.5-6.9 22.5-7l.3 11.5c-5.8.1-11.5 1.9-16.4 5.2-6.9 4.6-11.7 11.6-13.4 19.7-1.7 8.1-.1 16.5 4.5 23.4l-9.6 6.2zm99 14.1l-9.9-5.9c6-10 9-21.5 8.7-33.3-.8-33.8-28.6-60.7-62.2-60.7h-1.5L95.9 24c40.7-1 74.5 31.3 75.5 72 .3 13.8-3.3 27.4-10.3 39.3m-63.4 63.9c-35.2 0-68.3-18.5-86.7-48.7l9.8-6c16.7 27.4 47 43.9 79 43.1 31.1-.7 59.3-17.2 75.3-44l9.9 5.9c-18 30.3-49.7 48.8-84.9 49.6-.8.1-1.6.1-2.4.1"/></svg>'
                        },
                        title: "Marketing Cloud",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/digital-marketing.html?promoid=KLXLZ",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/digital-marketing.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "marketingCloudWebExperienceManager",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg version="1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 792 792" enable-background="new 0 0 792 792"><path d="M396.1.7C177.3.7 0 177.7 0 396s177.3 395.3 396 395.3 396-177 396-395.3C792.1 177.7 614.8.7 396.1.7zm0 29.1C598.4 29.8 763 194.1 763 396S598.4 762.2 396.1 762.2C193.7 762.2 29.2 597.9 29.2 396S193.7 29.8 396.1 29.8z" fill="#63A647"/><path d="M396 135.2c32.9 0 61.4 18.9 75.1 46.2-7.5 4.4-14.5 9.7-20.8 16-14.5 14.4-33.8 22.3-54.3 22.3s-39.8-7.9-54.2-22.2c-6.3-6.3-13.3-11.6-20.8-16 13.7-27.5 42.2-46.3 75-46.3zm83.2 73.6c.3 3 .5 6 .5 9.1 0 53.2 43.7 96.4 97.4 96.4 3.1 0 6.2.2 9.2.5-2.4 3.3-5 6.4-8 9.2-19.7 19.5-30.6 45.4-30.6 73s10.8 53.5 30.6 73c2.9 2.9 5.6 6 8 9.3-3 .3-6.1.5-9.2.5-53.7 0-97.4 43.3-97.4 96.4 0 3.1-.2 6.1-.5 9.1-3.3-2.4-6.4-5-9.4-7.9-19.7-19.7-45.9-30.4-73.8-30.4s-54.1 10.7-73.8 30.2c-2.9 2.9-6.1 5.5-9.4 7.9-.3-3-.5-6-.5-9.1 0-53.2-43.7-96.4-97.4-96.4-3.1 0-6.2-.2-9.2-.5 2.4-3.3 5-6.3 8-9.2 19.7-19.5 30.6-45.4 30.6-73s-10.9-53.5-30.6-73c-2.9-2.9-5.6-6-8-9.2 3-.3 6.1-.5 9.2-.5 53.7 0 97.4-43.3 97.4-96.4 0-3.1.2-6.1.5-9.1 3.3 2.4 6.4 5 9.4 7.9 19.7 19.5 45.9 30.3 73.8 30.3s54.1-10.8 73.8-30.3c3-2.8 6.1-5.4 9.4-7.8zm134.7 114c27.7 13.5 46.7 41.7 46.7 74.3 0 32.5-19.1 60.8-46.7 74.3-4.4-7.4-9.8-14.3-16.2-20.6-14.4-14.4-22.4-33.4-22.4-53.7s8-39.3 22.5-53.7c6.3-6.3 11.7-13.2 16.1-20.6zM471.1 612.7c-13.7 27.4-42.2 46.2-75.1 46.2s-61.4-18.9-75.1-46.2c7.5-4.4 14.5-9.7 20.8-16 14.5-14.3 33.8-22.2 54.3-22.2s39.8 7.9 54.3 22.2c6.3 6.3 13.3 11.6 20.8 16zM178.2 471.3c-27.7-13.5-46.7-41.7-46.7-74.2s19.1-60.7 46.7-74.3c4.4 7.4 9.8 14.3 16.2 20.6 14.5 14.4 22.5 33.4 22.5 53.7 0 20.3-8 39.4-22.5 53.7-6.4 6.2-11.8 13.1-16.2 20.5zm217.9 201.3c38.2 0 71.4-21.9 87.4-53.8 12.7 5.3 26.5 8.1 40.7 8.1 27.9 0 54.1-10.7 73.8-30.2s30.6-45.4 30.6-73c0-14.1-2.8-27.7-8.2-40.2 32.2-15.8 54.3-48.6 54.3-86.5s-22.2-70.7-54.3-86.5c5.4-12.6 8.2-26.2 8.2-40.2 0-27.6-10.9-53.5-30.5-73-19.7-19.5-45.9-30.3-73.8-30.3-14.2 0-28 2.8-40.7 8.1-15.9-31.8-49.1-54.1-87.4-54.1-38.3 0-71.4 22.3-87.3 54.1-12.7-5.3-26.5-8.1-40.6-8.1-27.9 0-54.1 10.7-73.8 30.2s-30.6 45.4-30.6 73c0 14.1 2.8 27.7 8.2 40.3-32.1 15.8-54.3 48.6-54.3 86.5s22.2 70.7 54.3 86.5c-5.4 12.6-8.2 26.2-8.2 40.2 0 27.6 10.9 53.5 30.6 73s45.9 30.3 73.8 30.3c14.2 0 28-2.8 40.6-8.1 15.7 31.8 48.9 53.7 87.2 53.7zm96-79.8c.9-5.4 1.4-10.9 1.4-16.6 0-45.6 37.5-82.8 83.6-82.8 5.7 0 11.3-.5 16.7-1.4 4.6 9.8 7 20.6 7 31.7 0 20.3-8 39.4-22.5 53.7-14.5 14.3-33.8 22.2-54.3 22.2-11.3 0-22.1-2.3-31.9-6.8zM593.8 302c-5.4-.9-11-1.4-16.7-1.4-46.1 0-83.6-37.1-83.6-82.8 0-5.6-.5-11.2-1.4-16.6 9.9-4.5 20.8-6.9 32-6.9 20.5 0 39.8 7.9 54.3 22.2 14.5 14.4 22.5 33.4 22.5 53.7-.2 11.4-2.6 22.1-7.1 31.8zM300.1 201.4c-.9 5.4-1.4 10.9-1.4 16.6 0 45.6-37.5 82.8-83.6 82.8-5.7 0-11.3.5-16.7 1.4-4.6-9.8-7-20.6-7-31.7 0-20.3 8-39.3 22.5-53.7 14.5-14.3 33.8-22.2 54.3-22.2 11.2-.1 22 2.3 31.9 6.8zM198.4 492.1c5.4.9 11 1.4 16.7 1.4 46.1 0 83.6 37.1 83.6 82.8 0 5.6.5 11.2 1.4 16.6-9.9 4.5-20.8 6.9-32 6.9-20.5 0-39.8-7.9-54.3-22.2s-22.5-33.4-22.5-53.7c.1-11.3 2.5-22.1 7.1-31.8z" fill="#ddd"/></svg>'
                        },
                        title: "Experience Manager",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/web-experience-management.html?promoid=KOUER",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/web-experience-management.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "marketingCloudAnalytics",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 90.2 612 611.5" enable-background="new 0 90.2 612 611.5"><path fill="#63A647" d="M306.1 90.2C137.6 90.2 1 227.2 1 396s136.6 305.8 305 305.8 305.1-137 305.1-305.8-136.5-305.8-305-305.8zm0 22.5c155.9 0 282.6 127.1 282.6 283.3S462 679.3 306.1 679.3 23.5 552.2 23.5 396s126.8-283.3 282.6-283.3z"/><path fill="#ddd" d="M232.5 401.8h-31.6v206.3c9.6 4.8 21.3 9 31.6 12.5V401.8zm58.2-13.2v243.1c4.4.2 11 2.5 15.5 2.5 5.6 0 10.7-.1 16.2-.5V388.6h-31.7zm181.2 172.5c11.2-11.8 23-26 31.6-39.9V363.8h-31.6v197.3zm-346-146.2l72.9-83.6 69.3 25.6 147.8-105.4 74.5 70.1 38.3-30.2-6.6-10.7-31.3 25-73.4-68.2L267 344l-69.4-26.5-77.6 87.3-48.7 11.3v11.1l54.6-12.3zm15.2 49.7h-31.6v60.9c8.7 13.3 20.4 25.7 31.6 37v-97.9zm272.6-138.9h-31.6v293.8c10.3-3.6 22-8 31.6-13.1V325.7z"/></svg>'
                        },
                        title: "Analytics",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/digital-analytics.html?promoid=KOUEP",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/digital-analytics.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "marketingCloudTesting",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg version="1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 792 792" enable-background="new 0 0 792 792"><path d="M396 .7C177.3.7 0 177.7 0 396s177.3 395.3 396 395.3 396-177 396-395.3S614.7.7 396 .7zm0 29.1c202.3 0 366.9 164.3 366.9 366.2S598.3 762.2 396 762.2 29.1 597.9 29.1 396 193.7 29.8 396 29.8z" fill="#63A647"/><path d="M211.5 316.5c21.7 0 42 8.4 57.3 23.6 15.3 15.3 23.8 35.5 23.8 57.1s-8.4 41.8-23.8 57.1c-15.3 15.2-35.7 23.6-57.3 23.6-21.7 0-42-8.4-57.3-23.7-15.3-15.2-23.8-35.5-23.8-57.1s8.5-41.8 23.8-57.1c15.3-15.1 35.6-23.5 57.3-23.5zM400.9 128c21.7 0 42 8.4 57.3 23.6 31.6 31.5 31.6 82.7 0 114.1-15.3 15.2-35.7 23.6-57.4 23.6s-42-8.4-57.3-23.6c-31.6-31.5-31.6-82.7 0-114.1 15.4-15.2 35.7-23.6 57.4-23.6zm-7.6 369.5c21.7 0 42 8.4 57.4 23.6 15.3 15.2 23.8 35.5 23.8 57.1s-8.4 41.8-23.8 57.1C435.4 650.5 415 659 393.3 659s-42-8.4-57.3-23.7c-31.6-31.5-31.6-82.7 0-114 15.3-15.4 35.7-23.8 57.3-23.8zm189.5-173.2c17.5 0 34.1 6.8 46.5 19.2 12.4 12.4 19.2 28.8 19.2 46.3s-6.8 33.9-19.2 46.3c-12.4 12.3-28.9 19.1-46.5 19.1s-34.1-6.8-46.5-19.1c-12.4-12.4-19.3-28.8-19.3-46.3s6.8-33.9 19.3-46.3c12.4-12.4 28.9-19.2 46.5-19.2zm-371.3-23.1c-24.7 0-49.4 9.4-68.2 28.1-37.7 37.5-37.7 98.3 0 135.7 18.8 18.7 43.5 28.1 68.2 28.1s49.4-9.4 68.2-28.1c35.8-35.6 37.6-92.3 5.3-129.9l55.1-53.5c16.1 13.7 35.6 21.2 55.4 22.7v178.2h-.5c-24.7 0-51.1 9.4-70 28.1-37.7 37.5-37.7 98 0 135.4 18.8 18.8 43.5 28.1 68.2 28.1s49.4-9.4 68.2-28.1c33.6-33.2 37.4-84.8 11.6-122.1l55.1-55.3c16.4 11.3 35.6 17 54.8 17 24.7 0 49.4-9.4 68.2-28.1 37.7-37.5 37.7-98.3 0-135.7-18.8-18.7-43.5-28.1-68.2-28.1s-49.4 9.4-68.2 28.1c-37.7 37.5-37.7 98.3 0 135.7l1.7 1.7-53.2 52.8-2.7-1.7c-14.7-14.6-33-23.6-52-26.8V304.2c21.7-1.9 42.9-11.2 59.5-27.7 37.7-37.5 38.8-98.3 1.1-135.7-18.8-18.7-43.5-28.1-68.2-28.1s-49.4 9.4-68.2 28.1c-35.8 35.6-37.6 92.3-5.4 129.9L273.9 324c-17.9-15.2-40.2-22.8-62.4-22.8zm414 86.7c0 22.6-14.8 42.6-41.4 43.1-28.1.5-43.5-18.5-43.5-41.1s19.2-43.5 41.9-43.5c22.8 0 43 18.9 43 41.5z" fill="#ddd"/></svg>'
                        },
                        title: "Target",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/testing-targeting.html?promoid=KOUEU",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/testing-targeting.html"
                        }
                    }
                }, {
                    hideForCountries: ["CN"],
                    iconAndText: {
                        id: "marketingCloudSocial",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg version="1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 792 792" enable-background="new 0 0 792 792"><path fill="#63A647" d="M396 .3C177.3.3 0 177.5 0 396s177.3 395.7 396 395.7S792 614.5 792 396 614.7.3 396 .3zm0 29.1c202.3 0 366.9 164.5 366.9 366.6S598.3 762.6 396 762.6 29.1 598.1 29.1 396 193.7 29.4 396 29.4z"/><path fill="#ddd" d="M220.2 534.4c.3-1.9.5-3.8.5-5.7 0-4.4-.7-8.7-2-12.7l234.6-119.8c6.7 9.1 17 15.3 28.8 16.5l9.9 174.6c-12.4 2.7-22.6 10.9-28 21.9l-243.8-74.8zM366.6 177c7.4 10.7 19.7 17.7 33.7 17.7 3 0 6-.4 8.8-1l56.4 143.4c-6.5 3.8-11.9 9.3-15.4 15.9l-240.4-82.7c.4-2.3.7-4.7.7-7.1 0-4-.6-7.8-1.7-11.5L366.6 177zm195.9 42.5c-.5 2.5-.8 5.1-.8 7.8 0 10.6 4.1 20.1 10.7 27.3l-66 82.1c-6-3.4-12.9-5.4-20.2-5.4-3 0-5.9.4-8.8 1L421 189c6.3-3.7 11.5-9 15.1-15.4l126.4 45.9zM634 430.9c-6.5 2.3-12.2 6.2-16.8 11.2l-92.7-55.8c1.6-4.4 2.6-9.1 2.6-14.1 0-10.6-4.1-20.2-10.7-27.4l66-82.1c6 3.4 12.9 5.4 20.2 5.4.4 0 .7 0 1.1-.1L634 430.9zm-449.4 57.3l-7.2-185.1c12.2-2.4 22.4-10.2 28.1-20.8L445.9 365c-.4 2.3-.7 4.7-.7 7.2 0 4.4.7 8.7 2 12.7L212.7 504.7c-6.5-8.9-16.6-15.2-28.1-16.5zm320.1 98.4l-10-174.6c8.6-1.8 16.1-6.3 21.7-12.6l92.7 55.8c-1.6 4.4-2.6 9.1-2.6 14.1 0 9.5 3.3 18.2 8.8 25.1l-92.1 98.8c-5.3-3.5-11.7-5.9-18.5-6.6zm142.9-158.1c-.4 0-.7 0-1 .1l-30.3-163c15.9-5.6 27.2-20.6 27.2-38.3 0-22.5-18.3-40.7-40.9-40.7-15.4 0-28.8 8.5-35.7 21l-126.4-45.8c.5-2.5.8-5.1.8-7.8 0-22.5-18.3-40.7-40.9-40.7-22.5 0-40.8 18.2-40.8 40.7 0 4 .6 7.8 1.7 11.5l-158 74.7c-7.4-10.7-19.7-17.7-33.7-17.7-22.6 0-40.9 18.2-40.9 40.7 0 20.8 15.7 38 36 40.4l7.1 185.1c-18.8 3.7-32.9 20.1-32.9 39.9 0 22.5 18.3 40.7 40.9 40.7 16.1 0 30-9.3 36.6-22.8l243.7 74.8c-.3 1.9-.5 3.8-.5 5.8 0 22.5 18.3 40.7 40.9 40.7 22.6 0 40.8-18.2 40.8-40.7 0-9.5-3.3-18.2-8.7-25.1l92.1-99c6.5 4.4 14.3 6.9 22.8 6.9 22.6 0 40.9-18.2 40.9-40.7.1-22.4-18.2-40.7-40.8-40.7z"/></svg>'
                        },
                        title: "Social",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/social-marketing.html?promoid=KOUET",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/social-marketing.html"
                        }
                    }
                }, {
                    hideForCountries: ["CN"],
                    iconAndText: {
                        id: "marketingCloudMedia",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg version="1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 792 792" enable-background="new 0 0 792 792"><path d="M396 1C177.3 1 0 177.9 0 396s177.3 395 396 395 396-176.9 396-395S614.6 1 396 1zm0 29.1c202.2 0 366.8 164.2 366.8 365.9S598.2 761.9 396 761.9 29.2 597.7 29.2 396 193.8 30.1 396 30.1z" fill="#63A647"/><path d="M613.4 183.9l-4.4-5.4L417.6 368V179.6h-17.4c-56.1.3-110.2 21.3-149 55.9l-59.1-60-5.5 5.3C129.1 237 96.5 312 95.9 392s31.1 155.3 87.7 212.2l5.6 5.4 196-189.4v189h16.3c56.1-.3 105.4-21.7 144.1-56.4l59.4 59.9 5.4-5.3c120-116.9 122.3-302.4 3-423.5zm-228.3 25v167.3H211c7.7-90.3 83-159.5 174.1-167.3zM192 197.1l48.7 48.5c-37.9 38.8-62.9 91.4-63 149.4v13.4h198.6L189.1 589C85.4 478.9 86.6 305.7 192 197.1zm225.6 385.7V408.4h171c-6.4 91.8-78.7 166.6-171 174.4zm187.4 8.4l-48.1-48.8c38.4-39.6 64-93.4 63.7-152.7v-13.3H429.8l178-176.4c103.9 110.1 102.6 282.4-2.8 391.2z" fill="#ddd"/></svg>'
                        },
                        title: "Media Optimizer",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/media-optimization.html?promoid=KOUES",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/media-optimization.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "marketingCloudCampaign",
                        itemVariants: [{
                            variant: "small-icon"
                        }
                        ],
                        staticIcon: {
                            icon: '<svg version="1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 792 792" enable-background="new 0 0 792 792"><path d="M396 1C177.3 1 0 177.9 0 396s177.4 395 396 395c218.7 0 396-176.9 396-395S614.7 1 396 1zm0 29.1c202.3 0 366.8 164.2 366.8 365.9S598.2 761.9 396 761.9 29.2 597.7 29.2 396 193.8 30.1 396 30.1z" fill="#63A647"/><g fill="#ddd"><path d="M397.8 470.9c-12.4 0-24.1-3.3-34.3-8.9v39.5c10.8 3.7 22.3 5.7 34.3 5.7 58.8 0 106.6-43.2 106.6-106.6l.3-289.2c-11.8-4.5-23.9-8.3-36.2-11.3l-.3 300.4c0 43.8-31.6 70.4-70.4 70.4zM397.8 330.1c12.5 0 24.3 3.3 34.5 9.1v-39.5c-10.8-3.7-22.4-5.7-34.5-5.7-58.8 0-106.6 47.8-106.6 106.6 0 15.6-.3 285.7-.3 285.7 11.8 4.3 23.9 7.9 36.2 10.7l.3-296.5c0-38.9 31.5-70.4 70.4-70.4zM544.6 400.5c0 91.3-65.8 147-146.8 147-11.8 0-23.3-1.4-34.3-4.1v16.3c11.1 2.4 22.5 3.7 34.3 3.7 89.8 0 162.7-64.8 162.7-162.8 0-2 .3-261.2.3-261.2-5.2-3.3-10.5-6.5-15.8-9.4-.1-.2-.4 270.4-.4 270.5zM397.8 253.5c11.9 0 23.4 1.4 34.5 4.1v-16.3c-11.1-2.4-22.6-3.7-34.5-3.7-89.8 0-162.8 73-162.8 162.8 0 7.9-.3 258.4-.3 258.4 5.2 3.2 10.5 6.3 15.8 9.2l.3-267.7c0-80.9 65.9-146.8 147-146.8zM598.9 400.5c0 117.7-90.2 201.2-201.1 201.2-11.7 0-23.1-1.1-34.3-3v16.1c11.2 1.8 22.6 2.7 34.3 2.7 119.6 0 216.9-90.4 216.9-217 0-1.3.7-217.3.7-217.3-5.1-5.2-10.4-10.2-15.8-15-.1 0-.7 232.2-.7 232.3zM180.4 616.2c5.1 5 10.4 9.9 15.8 14.6l.3-232c0-117.1 90.3-199.4 201.2-199.4 11.7 0 23.2 1.1 34.5 3v-16.1c-11.2-1.8-22.7-2.8-34.5-2.8-119.6 0-217 84.8-217 215.4 0 8.3-.3 217.3-.3 217.3z"/></g></svg>'
                        },
                        title: "Campaign",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/campaign-management.html?promoid=KOUEQ",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/campaign-management.html"
                        }
                    }
                }
                ]
            }, {
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "border"
                }, {
                    variant: "top-spacing-2"
                }, {
                    variant: "bottom-spacing-2"
                }, {
                    hideForCountries: ["JP"],
                    variant: "align-with-leading-rows"
                }, {
                    showForCountries: ["JP"],
                    variant: "tighten-horizontal-spacing-1"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "acrobat",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><path stroke="#000" stroke-width=".98" stroke-miterlimit="2.613" d="M94.6 399.5c-55 41.1-93 99.6-78 108.7l-12.9-6.6c-7.7-9.5 9.4-58.9 90.9-102.1zM508 353.5c28.2-59.9-203.8-50.9-373 27.5 131.6-50.8 375.8-72.8 373-27.5zM245.7 33.8c6.3-33.4-10.1-32-14.6-32h-10.4c-5.6 0-10.8 4.5-13.2 13.9-16 59.9 13.2 212.1 94.7 282.5 72.1 62 197.8 90.6 203.1 60.6-26.2 12.5-124.4-19.2-190.3-71.8-77.7-64.1-111.1-221.9-94.4-271.3 1.7-4.9 5.6-10.8 7.7-11.8 7 3.4 15.3 12.1 17.4 29.9zM244.3 43.5C237 79.1 233.9 162.3 166 299.9 92.5 447.2 39.2 515.8 3.7 501.6l12.9 6.6c27.9 14.3 79.1-32.7 164.4-196.8 65.1-126.5 61.6-191.6 63.3-267.9z"/></svg>'
                        },
                        title: "Acrobat",
                        url: {
                            defaultLink: "https://www.adobe.com/products/acrobat.html?promoid=KLXMA",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/acrobat.html"
                        }
                    }
                }, {
                    hideForCountries: ["CN", "HK", "IN", "JP", "MENA", "TR", "TW", "UA"],
                    iconAndText: {
                        id: "echoSign",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 57.88 59.409" enable-background="new 0 0 57.88 59.409"><path d="M15.787 38.772c1.052 1.532 1.933 2.755 3.038 2.773 2.622.043 12.975-11.75 12.975-11.75l1.964 1.937-.91 1.2c-.165.213-.36.458-.596.738-1.704 2.232-4.52 6.452-5.03 8.26 1.484-.61 4.772-3.402 7.48-5.7.932-.795 1.92-1.636 2.96-2.496.374-.308 1.506-1.25 2.587-.36 1.11.92.516 2.012-.38 3.668-2.063 3.804-2.27 5.082-2.084 5.49.324.13.48.136 1.757-.94.508-.43 1.08-.92 1.777-1.35 3.338-2.072 5.307-.408 6.748.812.895.76 1.74 1.47 3.045 1.73 2.688.517 4.485-1.292 4.506-1.305l2.257 2.16c-.1.107-2.294 2.374-5.775 2.374-.5 0-1.03-.05-1.582-.157-2.1-.402-3.41-1.514-4.465-2.405-1.354-1.148-1.676-1.428-3.095-.538-.5.308-.968.695-1.41 1.076-1.22 1.026-2.89 2.433-5.168 1.314-.766-.38-1.31-1.013-1.575-1.83-.316-1.005-.172-2.168.237-3.43-4.508 3.78-6.772 5.44-8.777 5.1-.78-.12-1.44-.58-1.863-1.275-.366-.612-.4-1.423-.215-2.34-2.664 2.13-3.596 3.206-5.53 3.16-1.88-.045-4.025-2.23-4.665-3.626l1.79-2.292z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M37.98 55.934H7.777c-2.343 0-4.303-2.292-4.303-4.634V7.778c0-2.342 1.96-4.303 4.303-4.303h21.017l13.57 13.735v10.26h3.64V15.95L30.35 0H7.79C3.496 0 0 3.492 0 7.783v43.844c0 4.29 3.495 7.783 7.79 7.783H38.01c4.296 0 7.996-3.487 7.996-7.778h-3.64c0 2.34-2.042 4.302-4.385 4.302z"/></svg>'
                        },
                        title: "EchoSign",
                        url: {
                            defaultLink: "https://www.echosign.adobe.com/?promoid=KLXMB"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "elements",
                        itemVariants: [{
                            variant: "large"
                        }, {
                            variant: "nudge-from-end-3"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="27.6 115.6 550.4 535.4" enable-background="new 27.6 115.6 550.4 535.4"><g fill="#fff"><path d="M197.2 361.7c-1.5 7-2.2 13.3-2.2 20.8 0 25.4 8.8 48.8 23.5 67.3l1.7 1 51.2-89.1h-74.2zM305.3 344.2l-36.2-65.9c-30.2 10.2-54.6 35.9-66.4 65.4l1.1.7 101.5-.2zM375.3 300.9c-19.1-16.8-44.1-27.1-71.6-27.1-5.4 0-10.7.5-15.9 1.3l-.7 1.2 51.4 88.2 36.8-63.6zM232.2 464.2c19.1 16.7 43.5 27.9 70.9 27.9 5.6 0 12.9-1 18.3-1.8l.6-1.1-52.9-89.3-36.9 64.3zM338.9 489.3c30.2-10.4 55.5-33.7 67-63.3l3.2-8.5-109-.2 38.8 72zM562.6 291.4L254.3 119.6c-14.8-8.3-34-3.1-42.7 11.4L31.8 433.9c-8.6 14.6-3.5 33.2 11.3 41.5l308.2 171.8c14.8 8.2 34 3 42.6-11.5l179.9-302.9c8.6-14.6 3.6-33.3-11.2-41.4zM303.7 545.8c-90.2 0-163.3-73.1-163.3-163.3s73.1-163.3 163.3-163.3S467 292.2 467 382.5c0 90.2-73.1 163.3-163.3 163.3zM387 313.5l-51.4 86.9h75.1c1.4-7 1.5-11.8 1.5-19.2 0-25.6-8.9-47.9-23.7-66.4l-1.5-1.3z"/></g></svg>'
                        },
                        title: "Elements",
                        url: {
                            defaultLink: "https://www.adobe.com/products/elements-family.html?promoid=KQQSC",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/elements-family.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "digitalPublishing",
                        itemVariants: [{
                            variant: "large"
                        }, {
                            variant: "no-wrap"
                        }, {
                            variant: "wrap-on-small-screen"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36.494 56.538" enable-background="new 0 0 36.494 56.538"><path fill-rule="evenodd" clip-rule="evenodd" d="M23.307 48.183V25.92l-17.02-5.85v22.264l17.02 5.85zm2.607 3.494l-22.08-7.46V16.588l22.08 7.46v27.627zm.503-32.333L3.53 12.13c-.252-.077-.522-.12-.8-.12C1.22 12.01 0 13.226 0 14.726v29.842c0 2.21 1.33 3.933 3.79 4.64l22.366 7.212c.255.077 1.043.118 1.322.118 1.507 0 2.73-1.214 2.73-2.712V23.982c0-2.21-1.332-3.933-3.79-4.638zM6.577 10.69l21.534 6.607c2.637 1.05 4.55 3.393 4.55 5.9V48.41h3.834v-37.72H6.577zm-.22-2.3h23.85V0L6.357 8.39z"/></svg>'
                        },
                        title: "Digital Publishing",
                        url: {
                            defaultLink: "https://www.adobe.com/products/digital-publishing-suite-enterprise.html?promoid=KLXMC",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/digital-publishing-suite-enterprise.html"
                        }
                    }
                }, {
                    hideForCountries: ["JP"],
                    iconAndText: {
                        id: "primeTime",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 90 612 612" enable-background="new 0 90 612 612"><path d="M404.6 397.5L247.2 481V314l157.4 83.5zM164.8 632c0 17.5 10.2 34.7 29.2 34.7h232.2c17.5 0 32.8-17.2 32.8-34.7v-59.5H164.8V632zm393.4-424.3c33.4 0 53.8 25.2 53.8 58.6v252.9c0 33.4-20.4 53.4-53.8 53.4h-63.9v69c0 33.4-25.2 60.4-58.6 60.4H184.5c-33.4 0-55-27-55-60.4v-69H62c-33.4 0-62-20-62-53.4V143.8C0 110.4 28.6 90 62 90h373.7c33.4 0 58.6 20.4 58.6 53.8v63.9h63.9zm-63.9 329.5h54.3c17.5 0 28-10.2 28-27.7V275.8c0-17.5-9-32.8-28-32.8h-54.3v294.2zM459 243H190.8c-17.4 0-26.1 15.3-26.1 32.8v261.4H459V243zM129.5 537.2v-271c0-33.4 21.6-58.6 55-58.6H459v-54.3c0-17.5-13.7-28-32.8-28H71.6c-19.1 0-36.2 10.6-36.2 28v356.2c0 17.5 17.2 27.7 36.2 27.7h57.9z"/></svg>'
                        },
                        title: "Primetime",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/primetime.html?promoid=KOUEV",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/primetime.html"
                        }
                    }
                }
                ]
            }, {
                groupVariants: [{
                    variant: "border"
                }, {
                    variant: "top-spacing-1"
                }, {
                    variant: "bottom-spacing-1"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "adobeReader",
                        itemVariants: [{
                            variant: "horizontal"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><path stroke="#000" stroke-width=".98" stroke-miterlimit="2.613" d="M94.6 399.5c-55 41.1-93 99.6-78 108.7l-12.9-6.6c-7.7-9.5 9.4-58.9 90.9-102.1zM508 353.5c28.2-59.9-203.8-50.9-373 27.5 131.6-50.8 375.8-72.8 373-27.5zM245.7 33.8c6.3-33.4-10.1-32-14.6-32h-10.4c-5.6 0-10.8 4.5-13.2 13.9-16 59.9 13.2 212.1 94.7 282.5 72.1 62 197.8 90.6 203.1 60.6-26.2 12.5-124.4-19.2-190.3-71.8-77.7-64.1-111.1-221.9-94.4-271.3 1.7-4.9 5.6-10.8 7.7-11.8 7 3.4 15.3 12.1 17.4 29.9zM244.3 43.5C237 79.1 233.9 162.3 166 299.9 92.5 447.2 39.2 515.8 3.7 501.6l12.9 6.6c27.9 14.3 79.1-32.7 164.4-196.8 65.1-126.5 61.6-191.6 63.3-267.9z"/></svg>'
                        },
                        title: "Adobe Reader",
                        url: {
                            defaultLink: "https://get.adobe.com/reader?promoid=KLXME",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://get.adobe.com/@@LOCALE@@/reader"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "adobeFlashPlayer",
                        itemVariants: [{
                            variant: "horizontal"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32.594 39.138" enable-background="new 0 0 32.594 39.138"><path fill="#231F20" d="M32.594 7.6l-.753.044c-4.853.28-7.52 3.748-9.54 8.085h6.014v7.63h-9.127c-1.25 3.09-2.898 6.18-4.937 9.035-3.32 4.647-8.12 6.504-13.42 6.71l-.83.033v-7.613l.753-.044c5.913-.336 8.75-4.895 12.005-13.275 1.444-4.28 3.036-7.81 5.72-11.497C21.355 2.75 26.26.23 31.763.032l.83-.032v7.6z"/></svg>'
                        },
                        title: "Adobe Flash Player",
                        url: {
                            defaultLink: "https://get.adobe.com/flashplayer?promoid=KLXMF",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://get.adobe.com/@@LOCALE@@/flashplayer"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "adobeAIR",
                        itemVariants: [{
                            variant: "horizontal"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 53.436 53.293" enable-background="new 0 0 53.436 53.293"><path d="M51.18 36.925c-.155-.21-.4-.506-.68-.79-2.89-2.96-5.05-6.457-6.213-10.328-.054-.18-.107-.36-.155-.542-.05-.182-.093-.368-.138-.55-.93-3.934-.814-8.042.215-12.048 0 0 .155-.662.196-.935.865-5.06-2.257-10.035-7.326-11.39-5.166-1.377-10.465 1.517-12.156 6.503-.07.186-.213.656-.214.664-1.113 3.925-3.045 7.493-5.773 10.413-3.1 3.244-7.052 5.5-11.454 6.735v.004c-5.38 1.44-8.576 6.962-7.14 12.327 1.44 5.362 6.97 8.546 12.35 7.104 4.02-1.025 8.138-1.143 12.08-.216.186.046.37.09.553.14.184.05.366.102.548.154 3.882 1.16 7.39 3.312 10.353 6.2.363.362.678.617.972.825 3.983 3.027 9.655 2.792 13.29-.836 3.69-3.675 3.907-9.486.69-13.435zm-12.374-4.852c-1.412 5.254-6.827 8.374-12.093 6.97-5.265-1.405-8.387-6.803-6.973-12.057 1.414-5.253 6.828-8.373 12.093-6.97 5.266 1.407 8.388 6.803 6.973 12.057z"/></svg>'
                        },
                        title: "Adobe AIR",
                        url: {
                            defaultLink: "https://get.adobe.com/air?promoid=KLXMG",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://get.adobe.com/@@LOCALE@@/air"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "adobeShockwavePlayer",
                        itemVariants: [{
                            variant: "horizontal"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 146" enable-background="new 0 0 192 146"><path d="M14.752 67.61c0 3.722-3.13 6.732-7 6.732s-7-3.01-7-6.73V7.07c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22 30c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V37.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22 14c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V51.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22 4c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V55.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22-6c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V49.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22-10c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V39.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22 0c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V39.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22 12c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V51.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54zm22 26c0 3.722-3.13 6.73-7 6.73s-7-3.01-7-6.73V77.073c0-3.72 3.13-6.73 7-6.73s7 3.01 7 6.73v60.54z"/></svg>'
                        },
                        title: "Adobe Shockwave Player",
                        url: {
                            defaultLink: "https://get.adobe.com/shockwave?promoid=KLXMH",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://get.adobe.com/@@LOCALE@@/shockwave"
                        }
                    }
                }
                ]
            }, {
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "border"
                }, {
                    variant: "top-spacing-2"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "allProducts",
                        itemVariants: [{
                            variant: "button"
                        }
                        ],
                        title: "All products",
                        url: {
                            defaultLink: "https://www.adobe.com/products/catalog/software._sl_id-contentfilter_sl_catalog_sl_software_sl_mostpopular.html?promoid=KLXMI",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/catalog/software._sl_id-contentfilter_sl_catalog_sl_software_sl_mostpopular_@@LOCALE@@.html",
                            localeCodeLinks: {
                                en_HK: "https://www.adobe.com/hk_en/products/catalog.html",
                                en_IN: "https://www.adobe.com/in/products/catalog.html",
                                en_MENA: "https://shop.adobe.com/store/adbehme/en_IE/home",
                                en_SEA: "https://www.adobe.com/sea/products/catalog.html"
                            }
                        }
                    }
                }
                ]
            }
            ]
        }, {
            id: "how-to-buy",
            groups: [{
                hideForCountries: ["CN", "RS"],
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "top-spacing-3"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "creativeCloud",
                        itemVariants: [{
                            variant: "creative-cloud"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" width="64.6" height="47" viewBox="22.7 16.5 64.6 47" enable-background="new 22.7 16.5 64.6 47"><path fill="#ccc" d="M85.1 34.3c-.2-.8-.5-1.7-.9-2.5.1.3.2.5.3.8-.2-.5-.5-1.1-.7-1.6.1.3.2.5.3.8-2.5-6-7.6-10.6-13.7-12.7-6.3-2.1-13.4-1.2-19.1 2.4-2.2 1.4-4.1 3.1-5.8 5.1-5.9-1-12.1 1-16.3 5.3-4.1 4.2-5.9 10.2-4.9 16C25.8 56.5 33.7 63 42.4 63h21c6.5 0 12.8-2.8 17-7.7 4.5-5.1 6.4-12 5.2-18.6-.1-.8-.3-1.7-.5-2.4 0-.2.5 2 0 0zm-31.4.3l6.7 7.5c1.5 1.6-1.1 3.8-2.5 2.3l-6.8-7.5C47 32.3 39.8 31.7 35 35.6s-5.7 11.1-2 16c2.2 2.9 5.8 4.7 9.4 4.7h4.9c1.3 1.3 2.7 2.4 4.2 3.3h-9.1c-6.2 0-12-4-14.1-9.8-2.1-5.7-.5-12.2 3.9-16.3 6.1-5.5 16-5 21.5 1.1zm2 23.5c-2.3-1-4.4-2.5-6.2-4.3l-8-8.1c-1.5-1.5.9-3.9 2.4-2.3l8 8.1c4.2 4.3 10.6 5.9 16.3 4.1 5.8-1.9 10-7 10.8-12.9.8-6-2-12-7-15.4-5.2-3.5-12.3-3.5-17.6 0-.8.5-1.5 1.1-2.3 1.8-1-.6-2-1.1-3.2-1.6 4.4-4.8 11.1-7.1 17.5-6 6 1 11.2 4.9 13.9 10.3 2.7 5.3 2.7 11.9 0 17.3s-8 9.3-13.9 10.3c-3.5.6-7.2.2-10.7-1.3-2.3-1 2.4 1 0 0z"/></svg>'
                        },
                        title: "Creative Cloud",
                        url: {
                            defaultLink: "https://www.adobe.com/creativecloud.html?promoid=KLXMJ#buy",
                            countryCodeLinks: {
                                HR: "https://www.adobe.com/hr/creativecloud.html#buy"
                            },
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud.html#buy",
                            excludeLocales: ["sr_RS", "zh_CN"]
                        }
                    }
                }
                ]
            }, {
                hideForCountries: ["CN", "RS"],
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "top-spacing-3"
                }, {
                    variant: "bottom-spacing-3"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "individuals",
                        itemVariants: [{
                            variant: "medium"
                        }
                        ],
                        hoverIcon: {
                            iconVariants: [{
                                variant: "large-and-wide"
                            }
                            ],
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 25.5 49.2 32" enable-background="new 0 25.5 49.2 32"><path stroke="#231F20" stroke-width="1.125" stroke-linejoin="round" stroke-miterlimit="10" d="M16 56.8h15.3s0-2.8-1.4-5.4c-1-2-5.3-3.4-10.7-5.3v-4s2-1.4 2-4c.7 0 1.4-2.7 0-3.4 0-.4 1.8-3.7 1.4-6-.7-3.4-10-3.4-10.7-.7-4 0-1.4 6.1-1.4 6.7-1.4.7-.7 3.4 0 3.4 0 2.7 2 4 2 4v4C7.3 48.2 3 49.4 2 51.5.6 54.2.6 56.9.6 56.9l15.4-.1z" fill="none"/><g stroke="#231F20" stroke-width="1.125" stroke-miterlimit="10" fill="none"><path d="M32.6 44.6v3.6l-6.7 2.4c-1.6.6-2.7 2.1-2.7 3.7v2.5h25.3v-2.5c0-1.7-1.1-3.2-2.7-3.7l-6.6-2.4v-3.9"/><ellipse cx="35.8" cy="37.5" rx="6.7" ry="8"/><path d="M42.4 36.8c-.2 0-.4 0-.6.1-2.3.4-3.7-.4-4.9-2.6-.8 1.5-3.1 2.6-5 2.6-1 0-1.8-.2-2.6-.6"/></g></svg>'
                        },
                        title: "Individuals",
                        url: {
                            defaultLink: "https://creative.adobe.com/plans?promoid=KLXMK",
                            languageCodeType: "defaultSet",
                            defaultLanguageLink: "https://creative.adobe.com/plans?locale=@@LANGUAGE@@",
                            localeCodeLinks: {
                                en_SEA: "https://creative.adobe.com/plans?locale=en&store_code=sg"
                            }
                        }
                    }
                }, {
                    iconAndText: {
                        id: "photographers",
                        itemVariants: [{
                            variant: "medium"
                        }, {
                            variant: "nudge-from-start-3"
                        }
                        ],
                        hoverIcon: {
                            iconVariants: [{
                                variant: "large"
                            }
                            ],
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50" enable-background="new 0 0 50 50"><g stroke="#231F20" stroke-width="2" stroke-miterlimit="10" fill="none"><path d="M5.2 19.8l.1-2.1c0-1.1.9-2.1 2-2.1h2c1.2 0 2.2 1 2.2 2.1v2.1"/><path d="M40.6 19.8l-2.1-4.2c-.7-1.1-1.8-2.1-3.1-2.1H25c-1.3 0-2.4 1-3.1 2.1l-3.1 4.2H5.2c-3.1 0-4.2 1.4-4.2 3.6V45c0 2.2 1 3.9 4.3 3.9h39.4c3.2 0 4.3-1.8 4.3-3.9V23.4c0-2.2-1-3.6-4.3-3.6h-4.1z"/><circle cx="30.2" cy="33.3" r="11.5"/><circle cx="30.2" cy="33.3" r="6.2"/><circle cx="8.3" cy="27.1" r="3.1"/></g></svg>'
                        },
                        title: "Photographers",
                        url: {
                            defaultLink: "https://creative.adobe.com/plans/photography?promoid=KLXML",
                            languageCodeType: "defaultSet",
                            defaultLanguageLink: "https://creative.adobe.com/plans/photography?locale=@@LANGUAGE@@",
                            localeCodeLinks: {
                                en_SEA: "https://creative.adobe.com/plans/photography?locale=en&store_code=sg"
                            }
                        }
                    }
                }, {
                    iconAndText: {
                        id: "studentsAndTeachers",
                        itemVariants: [{
                            variant: "medium"
                        }, {
                            variant: "nudge-from-start-3"
                        }
                        ],
                        hoverIcon: {
                            iconVariants: [{
                                variant: "large"
                            }
                            ],
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 46 42" enable-background="new 0 0 46 42"><g stroke="#231F20" stroke-width="2" stroke-miterlimit="10" fill="none"><path stroke-linecap="round" stroke-linejoin="round" d="M23 8c0-3.866 10.297-7 22-7v33c-11.703 0-22 3.134-22 7 0-3.866-10.3-7-22-7V1c11.7 0 22 3.134 22 7zM23 8v32M18 12.48C14.78 11.247 10.227 10.35 5 10M18 18.48c-3.22-1.232-7.773-2.127-13-2.48M18 24.48c-3.22-1.232-7.773-2.127-13-2.48M18 30.48c-3.22-1.232-7.773-2.127-13-2.48M28 24.48c3.223-1.233 7.773-2.128 13-2.48M28 30.48c3.223-1.233 7.773-2.128 13-2.48"/><path d="M41 1v17l-4-4-4 4V2"/></g></svg>'
                        },
                        title: "Students and Teachers",
                        url: {
                            defaultLink: "https://www.adobe.com/creativecloud/buy/students.html?promoid=KLXMM",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/buy/students.html",
                            excludeLocales: ["sr_RS", "zh_CN"],
                            localeCodeLinks: {
                                en_AFRICA: "https://www.adobe.com/africa/education.html",
                                en_EEUROPE: "https://www.adobe.com/eeurope/education.html"
                            }
                        }
                    }
                }, {
                    iconAndText: {
                        id: "smallAndMediumBusiness",
                        itemVariants: [{
                            variant: "medium"
                        }
                        ],
                        hoverIcon: {
                            iconVariants: [{
                                variant: "large"
                            }
                            ],
                            icon: '<svg width="38" height="48" xmlns="http://www.w3.org/2000/svg"><g stroke="#231F20" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" fill="none"><path stroke-dasharray="null" d="M1 17h20v30H1V17zM37 47H21V17h-4V1h20v46zM27 41h6v6h-6v-6zM5 41h6v6H5v-6z"/><path stroke-miterlimit="10" d="M1 21h8M1 25h6M17 9h8M17 5h12M17 13h6M1 29h4"/></g></svg>'
                        },
                        title: "Small and Medium Business",
                        url: {
                            defaultLink: "https://www.adobe.com/creativecloud/buy/business.html?promoid=KLXMN",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/buy/business.html",
                            excludeLocales: ["sr_RS", "zh_CN"]
                        }
                    }
                }, {
                    iconAndText: {
                        id: "enterprise",
                        itemVariants: [{
                            variant: "medium"
                        }
                        ],
                        hoverIcon: {
                            iconVariants: [{
                                variant: "large"
                            }
                            ],
                            icon: '<svg width="48" height="48" xmlns="http://www.w3.org/2000/svg"><g stroke="#231F20" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"><path stroke-miterlimit="10" d="M1 47h46"/><path d="M5 9h38v38H5V9z" stroke-dasharray="null"/><path stroke-miterlimit="10" d="M1 9h46"/><path d="M7 5h34v4H7V5zM19 1h10v4H19V1zM21 39h6v8h-6v-8z" stroke-dasharray="null"/><path stroke-miterlimit="10" d="M11 41h6M31 41h6M11 33h10M27 33h10M11 27h10M27 27h10M11 21h10M27 21h10M11 15h10M27 15h10"/></g></svg>'
                        },
                        title: "Enterprise",
                        url: {
                            defaultLink: "https://www.adobe.com/creativecloud/buy/enterprise.html?promoid=KLXMO",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/buy/enterprise.html",
                            excludeLocales: ["sr_RS", "zh_CN"]
                        }
                    }
                }, {
                    iconAndText: {
                        id: "schoolsAndUniversities",
                        itemVariants: [{
                            variant: "medium"
                        }
                        ],
                        hoverIcon: {
                            iconVariants: [{
                                variant: "large"
                            }
                            ],
                            icon: '<svg width="48" height="48" xmlns="http://www.w3.org/2000/svg"><g stroke-linecap="round" stroke-linejoin="round" stroke-width="2" stroke="#231F20" fill="none"><path d="M5 39h38v4H5v-4z" stroke-dasharray="null"/><path d="M7 17h34v22H7V17z" stroke-dasharray="null"/><path stroke-miterlimit="10" d="M47 47H1"/><path stroke-miterlimit="10" d="M47 43H1"/><path stroke-miterlimit="10" d="M11 17v22"/><path stroke-miterlimit="10" d="M21 17v22"/><path stroke-miterlimit="10" d="M17 17v22"/><path stroke-miterlimit="10" d="M27 17v22"/><path stroke-miterlimit="10" d="M31 17v22"/><path stroke-miterlimit="10" d="M37 17v22"/><path stroke-miterlimit="10" d="M45 13H3L24 1z"/><path d="M3 13h42v4H3v-4z" stroke-dasharray="null"/></g></svg>'
                        },
                        title: "Schools and Universities",
                        url: {
                            defaultLink: "https://www.adobe.com/creativecloud/buy/education.html?promoid=KLXMP",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/creativecloud/buy/education.html",
                            excludeLocales: ["sr_RS", "zh_CN"]
                        }
                    }
                }
                ]
            }, {
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "border",
                    hideForCountries: ["CN", "RS"]
                }, {
                    variant: "top-spacing-2"
                }, {
                    variant: "bottom-spacing-4"
                }, {
                    variant: "top-spacing-5",
                    showForCountries: ["CN", "RS"]
                }, {
                    variant: "tighten-horizontal-spacing-1"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "marketingCloud",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 196 199.2" enable-background="new 0 0 196 199.2"><path d="M97.7 136.5c-13.5 0-26.1-7.1-33.2-18.6l3.3-2c6.4 10.4 17.7 16.8 29.9 16.8h.8c12.1-.3 23-6.7 29.2-17.1l1.7 1-1.7-1c3.4-5.6 5-12.1 4.9-18.7-.2-9.4-4.1-18-10.8-24.5-6.8-6.4-15.6-9.9-25-9.6l-.1-3.8c10.4-.2 20.2 3.6 27.7 10.7 7.5 7.2 11.8 16.8 12 27.2.2 7.3-1.7 14.5-5.4 20.7-6.9 11.6-19 18.6-32.5 19l-.8-.1zm-81.6 15C6 136.2.5 118.4 0 100c-.6-26.1 9-50.9 27-69.7C45 11.4 69.3.7 95.4 0c26.1-.6 50.8 8.9 69.7 27 18.9 18 29.6 42.3 30.2 68.4.4 18.4-4.3 36.5-13.7 52.2l-3.3-2c9-15.1 13.5-32.5 13.1-50.1-.6-25.1-10.9-48.4-29.1-65.7-18.1-17.3-41.9-26.5-67-25.9l-.1-2.3.1 2.3c-25.1.6-48.4 10.9-65.7 29C12.5 51 3.3 74.8 3.9 99.9c.4 17.7 5.8 34.8 15.5 49.5l-3.3 2.1zm81.6 16.1c-23.4 0-45.4-11.9-58.4-31.4-7.2-10.9-11.2-23.7-11.5-36.9-.9-38.5 29.7-70.6 68.2-71.5l.1 3.8c-36.4.9-65.3 31.2-64.5 67.6.3 11.7 3.6 22.9 9.6 32.8 12.3 20.1 34.5 32.2 58 31.7 22.9-.5 43.5-12.6 55.2-32.3l3.3 2c-12.4 20.8-34.2 33.6-58.4 34.2h-1.6M62.1 121.2c-6.3-9.5-8.5-20.9-6.2-32 2.3-11.2 8.8-20.8 18.3-27 6.7-4.4 14.5-6.9 22.5-7l.3 11.5c-5.8.1-11.5 1.9-16.4 5.2-6.9 4.6-11.7 11.6-13.4 19.7-1.7 8.1-.1 16.5 4.5 23.4l-9.6 6.2zm99 14.1l-9.9-5.9c6-10 9-21.5 8.7-33.3-.8-33.8-28.6-60.7-62.2-60.7h-1.5L95.9 24c40.7-1 74.5 31.3 75.5 72 .3 13.8-3.3 27.4-10.3 39.3m-63.4 63.9c-35.2 0-68.3-18.5-86.7-48.7l9.8-6c16.7 27.4 47 43.9 79 43.1 31.1-.7 59.3-17.2 75.3-44l9.9 5.9c-18 30.3-49.7 48.8-84.9 49.6-.8.1-1.6.1-2.4.1"/></svg>'
                        },
                        title: "Marketing Cloud",
                        url: {
                            defaultLink: "https://www.adobe.com/solutions/digital-marketing.html?promoid=KLXMQ",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/solutions/digital-marketing.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "acrobat",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512"><path stroke="#000" stroke-width=".98" stroke-miterlimit="2.613" d="M94.6 399.5c-55 41.1-93 99.6-78 108.7l-12.9-6.6c-7.7-9.5 9.4-58.9 90.9-102.1zM508 353.5c28.2-59.9-203.8-50.9-373 27.5 131.6-50.8 375.8-72.8 373-27.5zM245.7 33.8c6.3-33.4-10.1-32-14.6-32h-10.4c-5.6 0-10.8 4.5-13.2 13.9-16 59.9 13.2 212.1 94.7 282.5 72.1 62 197.8 90.6 203.1 60.6-26.2 12.5-124.4-19.2-190.3-71.8-77.7-64.1-111.1-221.9-94.4-271.3 1.7-4.9 5.6-10.8 7.7-11.8 7 3.4 15.3 12.1 17.4 29.9zM244.3 43.5C237 79.1 233.9 162.3 166 299.9 92.5 447.2 39.2 515.8 3.7 501.6l12.9 6.6c27.9 14.3 79.1-32.7 164.4-196.8 65.1-126.5 61.6-191.6 63.3-267.9z"/></svg>'
                        },
                        title: "Acrobat",
                        url: {
                            defaultLink: "https://www.adobe.com/products/acrobatpro/buying-guide.html?promoid=KLXMR",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/acrobatpro/buying-guide.html"
                        }
                    }
                }, {
                    hideForCountries: ["CN", "HK", "IN", "JP", "MENA", "TR", "TW", "UA"],
                    iconAndText: {
                        id: "echoSign",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 57.88 59.409" enable-background="new 0 0 57.88 59.409"><path d="M15.787 38.772c1.052 1.532 1.933 2.755 3.038 2.773 2.622.043 12.975-11.75 12.975-11.75l1.964 1.937-.91 1.2c-.165.213-.36.458-.596.738-1.704 2.232-4.52 6.452-5.03 8.26 1.484-.61 4.772-3.402 7.48-5.7.932-.795 1.92-1.636 2.96-2.496.374-.308 1.506-1.25 2.587-.36 1.11.92.516 2.012-.38 3.668-2.063 3.804-2.27 5.082-2.084 5.49.324.13.48.136 1.757-.94.508-.43 1.08-.92 1.777-1.35 3.338-2.072 5.307-.408 6.748.812.895.76 1.74 1.47 3.045 1.73 2.688.517 4.485-1.292 4.506-1.305l2.257 2.16c-.1.107-2.294 2.374-5.775 2.374-.5 0-1.03-.05-1.582-.157-2.1-.402-3.41-1.514-4.465-2.405-1.354-1.148-1.676-1.428-3.095-.538-.5.308-.968.695-1.41 1.076-1.22 1.026-2.89 2.433-5.168 1.314-.766-.38-1.31-1.013-1.575-1.83-.316-1.005-.172-2.168.237-3.43-4.508 3.78-6.772 5.44-8.777 5.1-.78-.12-1.44-.58-1.863-1.275-.366-.612-.4-1.423-.215-2.34-2.664 2.13-3.596 3.206-5.53 3.16-1.88-.045-4.025-2.23-4.665-3.626l1.79-2.292z"/><path fill-rule="evenodd" clip-rule="evenodd" d="M37.98 55.934H7.777c-2.343 0-4.303-2.292-4.303-4.634V7.778c0-2.342 1.96-4.303 4.303-4.303h21.017l13.57 13.735v10.26h3.64V15.95L30.35 0H7.79C3.496 0 0 3.492 0 7.783v43.844c0 4.29 3.495 7.783 7.79 7.783H38.01c4.296 0 7.996-3.487 7.996-7.778h-3.64c0 2.34-2.042 4.302-4.385 4.302z"/></svg>'
                        },
                        title: "EchoSign",
                        url: {
                            defaultLink: "https://www.echosign.adobe.com/en/pricing.html?cs=mktg_topnav&promoid=KLXMS"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "elements",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="27.6 115.6 550.4 535.4" enable-background="new 27.6 115.6 550.4 535.4"><g fill="#fff"><path d="M197.2 361.7c-1.5 7-2.2 13.3-2.2 20.8 0 25.4 8.8 48.8 23.5 67.3l1.7 1 51.2-89.1h-74.2zM305.3 344.2l-36.2-65.9c-30.2 10.2-54.6 35.9-66.4 65.4l1.1.7 101.5-.2zM375.3 300.9c-19.1-16.8-44.1-27.1-71.6-27.1-5.4 0-10.7.5-15.9 1.3l-.7 1.2 51.4 88.2 36.8-63.6zM232.2 464.2c19.1 16.7 43.5 27.9 70.9 27.9 5.6 0 12.9-1 18.3-1.8l.6-1.1-52.9-89.3-36.9 64.3zM338.9 489.3c30.2-10.4 55.5-33.7 67-63.3l3.2-8.5-109-.2 38.8 72zM562.6 291.4L254.3 119.6c-14.8-8.3-34-3.1-42.7 11.4L31.8 433.9c-8.6 14.6-3.5 33.2 11.3 41.5l308.2 171.8c14.8 8.2 34 3 42.6-11.5l179.9-302.9c8.6-14.6 3.6-33.3-11.2-41.4zM303.7 545.8c-90.2 0-163.3-73.1-163.3-163.3s73.1-163.3 163.3-163.3S467 292.2 467 382.5c0 90.2-73.1 163.3-163.3 163.3zM387 313.5l-51.4 86.9h75.1c1.4-7 1.5-11.8 1.5-19.2 0-25.6-8.9-47.9-23.7-66.4l-1.5-1.3z"/></g></svg>'
                        },
                        title: "Elements",
                        url: {
                            defaultLink: "https://www.adobe.com/products/elements-family.html?promoid=KQQSD",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/elements-family.html"
                        }
                    }
                }, {
                    iconAndText: {
                        id: "digitalPublishing",
                        itemVariants: [{
                            variant: "large"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36.494 56.538" enable-background="new 0 0 36.494 56.538"><path fill-rule="evenodd" clip-rule="evenodd" d="M23.307 48.183V25.92l-17.02-5.85v22.264l17.02 5.85zm2.607 3.494l-22.08-7.46V16.588l22.08 7.46v27.627zm.503-32.333L3.53 12.13c-.252-.077-.522-.12-.8-.12C1.22 12.01 0 13.226 0 14.726v29.842c0 2.21 1.33 3.933 3.79 4.64l22.366 7.212c.255.077 1.043.118 1.322.118 1.507 0 2.73-1.214 2.73-2.712V23.982c0-2.21-1.332-3.933-3.79-4.638zM6.577 10.69l21.534 6.607c2.637 1.05 4.55 3.393 4.55 5.9V48.41h3.834v-37.72H6.577zm-.22-2.3h23.85V0L6.357 8.39z"/></svg>'
                        },
                        title: "Digital Publishing",
                        url: {
                            defaultLink: "https://www.adobe.com/products/digital-publishing-suite-enterprise.html?promoid=KLXMT",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/digital-publishing-suite-enterprise.html"
                        }
                    }
                }
                ]
            }, {
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "top-spacing-2",
                    showForCountries: ["CN", "RS"]
                }
                ],
                items: [{
                    iconAndText: {
                        id: "allProducts",
                        itemVariants: [{
                            variant: "button"
                        }
                        ],
                        title: "All products",
                        url: {
                            defaultLink: "https://www.adobe.com/products/catalog/software._sl_id-contentfilter_sl_catalog_sl_software_sl_mostpopular.html?promoid=KLXMV",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/products/catalog/software._sl_id-contentfilter_sl_catalog_sl_software_sl_mostpopular_@@LOCALE@@.html",
                            localeCodeLinks: {
                                en_HK: "https://www.adobe.com/hk_en/products/catalog.html",
                                en_IN: "https://www.adobe.com/in/products/catalog.html",
                                en_MENA: "https://shop.adobe.com/store/adbehme/en_IE/home",
                                en_SEA: "https://www.adobe.com/sea/products/catalog.html"
                            }
                        }
                    }
                }
                ]
            }
            ]
        }, {
            id: "learn-and-support",
            groups: [{
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "top-spacing-5"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "learnAtYourLevel",
                        itemVariants: [{
                            variant: "large"
                        }, {
                            variant: "large-on-small-screen"
                        }, {
                            variant: "horizontal-spacing-1"
                        }, {
                            variant: "top-spacing-on-small-screen-2"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg width="48" height="48" xmlns="http://www.w3.org/2000/svg"><g stroke-linecap="round" stroke-linejoin="round" stroke-width="2" stroke="#231F20" fill="none"><path d="M1 7h46v34H1V7zM1 1h46v6H1V1z" stroke-dasharray="null"/><path stroke-miterlimit="10" d="M33 1v6M15 1v6"/><path d="M1 41h46v6H1v-6z" stroke-dasharray="null"/><path stroke-miterlimit="10" d="M33 41v6M15 41v6"/><circle stroke-miterlimit="10" cx="24" cy="24" r="11"/><path stroke-miterlimit="10" d="M21 28v-8l8 4z"/></g></svg>'
                        },
                        title: "Learn at your level",
                        url: {
                            defaultLink: "https://helpx.adobe.com/support.html?promoid=KLXMW",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://helpx.adobe.com/@@LOCALE@@/support.html",
                            localeCodeLinks: {
                                en_CA: "https://helpx.adobe.com/support.html"
                            }
                        },
                        itemDescription: {
                            description: "Get started or go deeper with tutorials of all our products."
                        },
                        itemCallToAction: {
                            title: "Learn now >",
                            url: {
                                defaultLink: "https://helpx.adobe.com/support.html?promoid=KLXMZ",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://helpx.adobe.com/@@LOCALE@@/support.html",
                                localeCodeLinks: {
                                    en_CA: "https://helpx.adobe.com/support.html"
                                }
                            }
                        }
                    }
                }, {
                    iconAndText: {
                        id: "contactSupport",
                        itemVariants: [{
                            variant: "large"
                        }, {
                            variant: "large-on-small-screen"
                        }, {
                            variant: "horizontal-spacing-1"
                        }, {
                            variant: "top-spacing-on-small-screen-2"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" enable-background="new 0 0 48 48"><path stroke="#231F20" stroke-width="2" stroke-miterlimit="10" d="M12 32.396v4.34l-7.883 2.825C2.23 40.236 1 42.02 1 44.023V47h30v-2.978c0-2.002-1.232-3.787-3.117-4.46L20 36.736v-4.612" fill="none"/><ellipse stroke="#231F20" stroke-width="2" stroke-miterlimit="10" cx="15.802" cy="24.104" rx="7.895" ry="9.475" fill="none"/><path stroke="#231F20" stroke-width="2" stroke-miterlimit="10" d="M23.646 23.25c-.265.032-.47.015-.738.065-2.69.52-4.416-.456-5.897-3.047-.887 1.703-3.663 3.047-5.944 3.047-1.122 0-2.113-.205-3.08-.712M21 13c0-6.628 5.82-12 13-12s13 5.372 13 12c0 4.992-3.302 9.272-8 11.08C38 25 33 30 32 31c0-1 0-5 .007-6.14-1.06-.15-2.08-.42-3.04-.792" fill="none"/><path fill="#231F20" d="M40 13h-2v2h2v-2zM35 13h-2v2h2v-2zM30 13h-2v2h2v-2z"/></svg>'
                        },
                        title: "Contact support",
                        url: {
                            defaultLink: "https://helpx.adobe.com/contact.html?promoid=KLXMX",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://helpx.adobe.com/@@LOCALE@@/contact.html",
                            localeCodeLinks: {
                                en_CA: "https://helpx.adobe.com/contact.html"
                            }
                        },
                        itemDescription: {
                            description: "Get instant help from one of our awesome support people."
                        },
                        itemCallToAction: {
                            title: "Start now >",
                            url: {
                                defaultLink: "https://helpx.adobe.com/contact.html?promoid=KLXNA",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://helpx.adobe.com/@@LOCALE@@/contact.html",
                                localeCodeLinks: {
                                    en_CA: "https://helpx.adobe.com/contact.html"
                                }
                            }
                        }
                    }
                }, {
                    iconAndText: {
                        id: "askTheCommunity",
                        itemVariants: [{
                            variant: "large"
                        }, {
                            variant: "large-on-small-screen"
                        }, {
                            variant: "horizontal-spacing-1"
                        }, {
                            variant: "top-spacing-on-small-screen-2"
                        }
                        ],
                        hoverIcon: {
                            icon: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 47" enable-background="new 0 0 48 47"><g stroke="#231F20" stroke-width="2" stroke-linejoin="round" stroke-miterlimit="10" fill="none"><path d="M14.88 28.628c-.978-.134-1.943-.342-2.88-.628L2 32l4-7c-3.147-2.518-5-6.25-5-10.13C1 7.21 8.61 1 18 1s17 6.21 17 13.87"/><path d="M19 32c0 8.576 10 15 19 11l8 3-3-6c2.592-2.074 4-4.806 4-8 0-6.31-6.267-12-14-12-7.732 0-14 5.69-14 12z"/></g></svg>'
                        },
                        title: "Ask the community",
                        url: {
                            defaultLink: "https://forums.adobe.com/welcome?promoid=KLXMY",
                            countryCodeLinks: {
                                DE: "https://forums.adobe.com/community/international_forums/deutsche",
                                ES: "https://forums.adobe.com/community/international_forums/espanol",
                                FR: "https://forums.adobe.com/community/international_forums/francais",
                                JP: "https://forums.adobe.com/community/international_forums/japanese?promoid=KKGUM",
                                MX: "https://forums.adobe.com/community/international_forums/espanol"
                            }
                        },
                        itemDescription: {
                            description: "Post, discuss, and be a part of our knowledgeable community."
                        },
                        itemCallToAction: {
                            title: "Join now >",
                            url: {
                                defaultLink: "https://forums.adobe.com/welcome?promoid=KLXNB",
                                countryCodeLinks: {
                                    DE: "https://forums.adobe.com/community/international_forums/deutsche",
                                    ES: "https://forums.adobe.com/community/international_forums/espanol",
                                    FR: "https://forums.adobe.com/community/international_forums/francais",
                                    JP: "https://forums.adobe.com/community/international_forums/japanese?promoid=KKGUM",
                                    MX: "https://forums.adobe.com/community/international_forums/espanol"
                                }
                            }
                        }
                    }
                }
                ]
            }, {
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "top-spacing-5"
                }
                ],
                items: [{
                    iconAndText: {
                        id: "allLearnAndSupport",
                        itemVariants: [{
                            variant: "button"
                        }, {
                            variant: "top-spacing-on-small-screen-2"
                        }
                        ],
                        title: "All learn & support",
                        url: {
                            defaultLink: "https://helpx.adobe.com/support.html?promoid=KLXNC",
                            localeCodeType: "defaultSet",
                            defaultLocaleLink: "https://helpx.adobe.com/@@LOCALE@@/support.html",
                            localeCodeLinks: {
                                en_CA: "https://helpx.adobe.com/support.html"
                            }
                        }
                    }
                }
                ]
            }
            ]
        }, {
            id: "about-adobe",
            sectionVariants: [{
                variant: "narrow"
            }
            ],
            groups: [{
                groupVariants: [{
                    variant: "top-spacing-3"
                }, {
                    variant: "bottom-spacing-3"
                }
                ],
                items: [{
                    megaIcon: {
                        itemVariants: [{
                            variant: "large-on-small-screen"
                        }
                        ],
                        staticIcon: '<svg xmlns="http://www.w3.org/2000/svg" width="36.5" height="32" viewBox="0 0 36.5 32"><g fill="#999"><path d="M35.68 32c-.193 0-.37-.114-.448-.298L22.225.672c-.062-.15-.047-.32.044-.456.09-.136.24-.216.402-.216H35.68c.267 0 .484.217.484.485v31.03c0 .23-.163.43-.39.475-.03.007-.063.01-.095.01zM23.4.97l11.792 28.135V.97H23.402zM.485 32c-.032 0-.063-.003-.096-.01-.227-.046-.39-.244-.39-.475V.485C0 .217.217 0 .485 0h13.02c.163 0 .315.08.405.216.09.136.106.306.043.456L.933 31.702c-.08.183-.256.298-.448.298zM.97.97v28.138L12.777.97H.97zM26.378 32h-5.434c-.2 0-.377-.12-.45-.305l-2.36-5.937h-5.738c-.163 0-.315-.08-.404-.218-.09-.136-.106-.307-.042-.458l5.69-13.352c.078-.178.253-.295.447-.295.195 0 .37.117.446.296l8.292 19.596c.062.15.047.322-.043.458s-.24.217-.404.217zm-5.107-.97h4.375l-7.56-17.87-4.956 11.627h5.33c.2 0 .378.12.452.305l2.358 5.937z"/></g></svg>'
                    }
                }, {
                    linkList: {
                        itemVariants: [{
                            variant: "large-on-small-screen"
                        }
                        ],
                        items: [{
                            id: "aboutUs",
                            title: "About Us",
                            url: {
                                defaultLink: "https://www.adobe.com/company.html?promoid=KLXND",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/company.html"
                            }
                        }, {
                            id: "careersAtAdobe",
                            title: "Careers At Adobe",
                            url: {
                                defaultLink: "https://www.adobe.com/careers.html?promoid=KLXNE",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/careers.html"
                            }
                        }, {
                            hideForCountries: ["JP"],
                            id: "investorRelations",
                            title: "Investor Relations",
                            url: {
                                defaultLink: "https://www.adobe.com/investor-relations.html?promoid=KLXNF"
                            }
                        }, {
                            id: "privacy",
                            title: "Privacy",
                            url: {
                                defaultLink: "https://www.adobe.com/privacy.html?promoid=KLXNG",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/privacy.html"
                            },
                            additionalLinks: [{
                                hasLinebreak: !1,
                                title: "Security",
                                url: {
                                    defaultLink: "https://www.adobe.com/security.html?promoid=KOVFG",
                                    localeCodeType: "defaultSet",
                                    defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/security.html"
                                }
                            }
                            ]
                        }, {
                            showForCountries: ["JP"],
                            id: "salesPartners",
                            title: "Sales Partners",
                            url: {
                                defaultLink: "https://www.adobe.com/jp/jos/elicensing/partners.html"
                            }
                        }
                        ]
                    }
                }, {
                    linkList: {
                        itemVariants: [{
                            variant: "large-on-small-screen"
                        }
                        ],
                        items: [{
                            id: "corporateResponsibility",
                            title: "Corporate Responsibility",
                            url: {
                                defaultLink: "https://www.adobe.com/corporate-responsibility.html?promoid=KLXNH",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/corporate-responsibility.html"
                            }
                        }, {
                            id: "customerShowcase",
                            title: "Customer Showcase",
                            url: {
                                defaultLink: "https://www.adobe.com/customershowcase.html?promoid=KLXNI",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/customershowcase.html"
                            }
                        }, {
                            id: "events",
                            title: "Events",
                            url: {
                                defaultLink: "https://www.adobe.com/events.html?promoid=KLXNJ",
                                localeCodeLinks: {
                                    en_AFRICA: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_AU: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=pa",
                                    en_BE: "https://www.adobe.com/events.html",
                                    en_CA: "https://www.adobe.com/events.html",
                                    en_CY: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_EEUROPE: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_GB: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_GR: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_HK: "http://www.adobe-gcevent.com/hk_en",
                                    en_IE: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_IL: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_IN: "https://www.adobe.com/events.html",
                                    en_LU: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_MENA: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_MT: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk",
                                    en_NZ: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=pa",
                                    en_SEA: "https://events.adobe.co.uk/cgi-bin/main.cgi?country=uk"
                                }
                            }
                        }, {
                            id: "contactUs",
                            title: "Contact Us",
                            url: {
                                defaultLink: "https://www.adobe.com/company/contact.html?promoid=KLXNK",
                                localeCodeType: "defaultSet",
                                defaultLocaleLink: "https://www.adobe.com/@@LOCALE@@/company/contact.html"
                            }
                        }
                        ]
                    }
                }
                ]
            }, {
                header: {
                    hoverTitle: {
                        title: "News",
                        url: {
                            defaultLink: "https://www.adobe.com/news-room.html?promoid=JZEGA",
                            localeCodeLinks: {
                                en_AFRICA: "https://www.adobe.com/africa/aboutadobe/pressroom/",
                                en_AU: "https://www.adobe.com/au/aboutadobe/pressroom/",
                                en_BE: "https://www.adobe.com/uk/aboutadobe/pressroom/",
                                en_CA: "https://www.adobe.com/news-room.html",
                                en_CY: "https://www.adobe.com/aboutadobe/pressroom/",
                                en_EEUROPE: "https://www.adobe.com/eeurope/aboutadobe/pressroom/",
                                en_GB: "https://www.adobe.com/uk/aboutadobe/pressroom/",
                                en_GR: "https://www.adobe.com/aboutadobe/pressroom/",
                                en_HK: "https://www.adobe.com/hk_en/aboutadobe/pressroom/",
                                en_IE: "https://www.adobe.com/uk/aboutadobe/pressroom/",
                                en_IL: "https://www.adobe.com/il_en/aboutadobe/pressroom/",
                                en_IN: "https://www.adobe.com/in/aboutadobe/pressroom/",
                                en_LU: "https://www.adobe.com/uk/aboutadobe/pressroom/",
                                en_MENA: "https://www.adobe.com/aboutadobe/pressroom/",
                                en_MT: "https://www.adobe.com/aboutadobe/pressroom/",
                                en_NZ: "https://www.adobe.com/nz/aboutadobe/pressroom/",
                                en_SEA: "https://www.adobe.com/sea/aboutadobe/pressroom/"
                            }
                        }
                    }
                },
                groupVariants: [{
                    variant: "equiwidth"
                }, {
                    variant: "border"
                }
                ],
                items: [{
                    news: {
                        items: [{
                            date: "10/27/2014",
                            title: "Adobe Partners with Workday to Modernize HR Processes with E-signatures",
                            url: {
                                defaultLink: "https://www.adobe.com/news-room/pressreleases/201410/102714AdobePartnersWithWorkdayToModernizeHRProcesses.html"
                            }
                        }, {
                            date: "10/21/2014",
                            title: "Adobe Named a Leader in Enterprise Marketing Software Suites by Independent Research Firm",
                            url: {
                                defaultLink: "https://www.adobe.com/news-room/pressreleases/201410/102114AdobeNamedALeaderInEnterpriseMarketingSoftwareSuites.html"
                            }
                        }, {
                            date: "10/21/2014",
                            title: "Adobe and Nielsen to Create Industry’s First Comprehensive Measurement Platform for Digital Content",
                            url: {
                                defaultLink: "http://www.businesswire.com/news/home/20141021005376/en"
                            }
                        }, {
                            date: "10/20/2014",
                            title: "Adobe Report Shows Online TV Consumption Up 388 Percent",
                            url: {
                                defaultLink: "http://www.businesswire.com/news/home/20141020006744/en"
                            }
                        }
                        ]
                    }
                }
                ]
            }
            ]
        }
        ]
    },
    search: {
        label: "Search",
        url: {
            defaultLink: "https://www.adobe.com/go/gnav_search"
        },
        loc: {
            defaultLink: "en_us",
            localeCodeLinks: {
                en_AFRICA: "africa",
                en_AU: "au",
                en_BE: "be_en",
                en_CA: "en_us",
                en_CY: "cy_en",
                en_EEUROPE: "eeurope",
                en_GB: "uk",
                en_GR: "gr_en",
                en_HK: "hk_en",
                en_IE: "ie",
                en_IL: "il_en",
                en_IN: "in",
                en_LU: "lu_en",
                en_MENA: "en_xem",
                en_MT: "mt",
                en_NZ: "nz",
                en_SEA: "sea"
            }
        },
        results: {
            seeAllSearchResults: "See all search results",
            tryAdvancedSearch: "Try our advanced search"
        }
    },
    all: {
        close: {
            icon: '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><path d="M31.158 32.118c-.246 0-.492-.093-.678-.282L.163 1.52c-.375-.375-.375-.983 0-1.358s.983-.375 1.358 0l30.317 30.316c.375.375.375.983 0 1.358-.187.188-.433.282-.678.282zM.842 32.118c-.246 0-.492-.093-.678-.282-.375-.375-.375-.983 0-1.358L30.48.162c.375-.375.983-.375 1.358 0s.375.983 0 1.358L1.52 31.836c-.186.188-.432.282-.677.282z" fill="#fff"/></svg>'
        },
        noJSNote: "For the complete experience, please enable JavaScript in your browser. Thank you!"
    }
});


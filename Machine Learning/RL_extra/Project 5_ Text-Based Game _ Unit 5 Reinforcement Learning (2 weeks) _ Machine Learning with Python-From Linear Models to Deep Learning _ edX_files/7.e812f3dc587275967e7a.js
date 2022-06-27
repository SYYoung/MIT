(window.webpackJsonp=window.webpackJsonp||[]).push([[7],{982:function(e,a,r){"use strict";r.r(a);var t=r(0),n=r.n(t),i=r(1),o=r.n(i),s=r(130),l=r(31),c=r(78),u=r(337),d=r(21),p=Object(l.g)({upgradeNow:{id:"learning.accessExpiration.upgradeNow",defaultMessage:"Upgrade now"}});function m(){return(m=Object.assign||function(e){for(var a=1;a<arguments.length;a++){var r=arguments[a];for(var t in r)Object.prototype.hasOwnProperty.call(r,t)&&(e[t]=r[t])}return e}).apply(this,arguments)}function g(e){var a=e.payload,r=a.accessExpiration,t=a.userTimezone,i=t?{timeZone:t}:{};if(!r)return null;var o=r.expirationDate,s=r.upgradeDeadline,c=r.upgradeUrl,g=null,y=function(e,a){return n.a.createElement(l.a,m({key:"accessExpiration.".concat(a),day:"numeric",month:"short",year:"numeric",value:e},i))};return s&&c&&(g=n.a.createElement(n.a.Fragment,null,"Upgrade by ",y(s,"upgradeDesc")," to unlock unlimited access to all course activities, including graded assignments.  ",n.a.createElement(u.a,{className:"font-weight-bold",style:{textDecoration:"underline"},destination:c},p.upgradeNow.defaultMessage))),n.a.createElement(d.b,{type:d.a.INFO},n.a.createElement("span",{className:"font-weight-bold"},"Unlock full course content by ",y(s,"upgradeTitle")),n.a.createElement("br",null),g,n.a.createElement("br",null),"You lose all access to the first two weeks of scheduled content on ",y(o,"expirationBody"),".")}g.propTypes={payload:o.a.shape({accessExpiration:o.a.shape({expirationDate:o.a.string.isRequired,masqueradingExpiredCourse:o.a.bool.isRequired,upgradeDeadline:o.a.string,upgradeUrl:o.a.string}).isRequired,userTimezone:o.a.string.isRequired}).isRequired};var y=Object(c.a)(g);function f(){return(f=Object.assign||function(e){for(var a=1;a<arguments.length;a++){var r=arguments[a];for(var t in r)Object.prototype.hasOwnProperty.call(r,t)&&(e[t]=r[t])}return e}).apply(this,arguments)}function h(e,a){return function(e){if(Array.isArray(e))return e}(e)||function(e,a){if("undefined"==typeof Symbol||!(Symbol.iterator in Object(e)))return;var r=[],t=!0,n=!1,i=void 0;try{for(var o,s=e[Symbol.iterator]();!(t=(o=s.next()).done)&&(r.push(o.value),!a||r.length!==a);t=!0);}catch(e){n=!0,i=e}finally{try{t||null==s.return||s.return()}finally{if(n)throw i}}return r}(e,a)||function(e,a){if(!e)return;if("string"==typeof e)return b(e,a);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(e);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return b(e,a)}(e,a)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}function b(e,a){(null==a||a>e.length)&&(a=e.length);for(var r=0,t=new Array(a);r<a;r++)t[r]=e[r];return t}function E(e){var a=e.intl,r=e.payload,i=h(Object(t.useState)(!!window.experiment__home_alert_bShowMMP2P),2),o=i[0],c=i[1];void 0===window.experiment__home_alert_showMMP2P&&(window.experiment__home_alert_showMMP2P=function(e){window.experiment__home_alert_bShowMMP2P=!!e,c(!!e)});var m=r.accessExpiration,g=r.courseId,b=r.org,E=r.userTimezone,x=r.analyticsPageName,w=E?{timeZone:E}:{};if(!m)return null;var v=m.expirationDate,_=m.masqueradingExpiredCourse,k=m.upgradeDeadline,q=m.upgradeUrl;if(_)return n.a.createElement(d.b,{type:d.a.INFO},n.a.createElement(l.b,{id:"learning.accessExpiration.expired",defaultMessage:"This learner does not have access to this course. Their access expired on {date}.",values:{date:n.a.createElement(l.a,f({key:"accessExpirationExpiredDate",day:"numeric",month:"short",year:"numeric",value:v},w))}}));if(o)return n.a.createElement(y,{payload:r});var O=null;return k&&q&&(O=n.a.createElement(n.a.Fragment,null,n.a.createElement("br",null),n.a.createElement(l.b,{id:"learning.accessExpiration.deadline",defaultMessage:"Upgrade by {date} to get unlimited access to the course as long as it exists on the site.",values:{date:n.a.createElement(l.a,f({key:"accessExpirationUpgradeDeadline",day:"numeric",month:"short",year:"numeric",value:k},w))}})," ",n.a.createElement(u.a,{className:"font-weight-bold",style:{textDecoration:"underline"},destination:q,onClick:function(){Object(s.e)("edx.bi.ecommerce.upsell_links_clicked",{org_key:b,courserun_key:g,linkCategory:"FBE_banner",linkName:"".concat(x,"_audit_access_expires"),linkType:"link",pageName:x})}},a.formatMessage(p.upgradeNow)))),n.a.createElement(d.b,{type:d.a.INFO},n.a.createElement("span",{className:"font-weight-bold"},n.a.createElement(l.b,{id:"learning.accessExpiration.header",defaultMessage:"Audit Access Expires {date}",values:{date:n.a.createElement(l.a,f({key:"accessExpirationHeaderDate",day:"numeric",month:"short",year:"numeric",value:v},w))}})),n.a.createElement("br",null),n.a.createElement(l.b,{id:"learning.accessExpiration.body",defaultMessage:"You lose all access to this course, including your progress, on {date}.",values:{date:n.a.createElement(l.a,f({key:"accessExpirationBodyDate",day:"numeric",month:"short",year:"numeric",value:v},w))}}),O)}E.propTypes={intl:l.i.isRequired,payload:o.a.shape({accessExpiration:o.a.shape({expirationDate:o.a.string.isRequired,masqueradingExpiredCourse:o.a.bool.isRequired,upgradeDeadline:o.a.string,upgradeUrl:o.a.string}).isRequired,courseId:o.a.string.isRequired,org:o.a.string.isRequired,userTimezone:o.a.string.isRequired,analyticsPageName:o.a.string.isRequired}).isRequired};a.default=Object(c.a)(E)}}]);
//# sourceMappingURL=7.e812f3dc587275967e7a.js.map
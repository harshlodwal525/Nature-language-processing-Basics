# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 19:57:02 2020

@author: hp
"""

import nltk

from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """I"Bad air quality" and "Air quality" redirect here. For the obsolete medical theory, see Bad air. For the measure of how polluted the air is, see Air quality index. For the properties of air, see Qualities of air.

Air pollution from a coking oven.

2016 air quality indicator - light colors have lower air quality and thus higher air pollution.
Air pollution is the presence of substances in the atmosphere that are harmful to the health of humans and other living beings, or cause damage to the climate or to materials. There are different types of air pollutants, such as gases (such as ammonia, carbon monoxide, sulfur dioxide, nitrous oxides, methane and chlorofluorocarbons), particulates (both organic and inorganic), and biological molecules. Air pollution may cause diseases, allergies and even death to humans; it may also cause harm to other living organisms such as animals and food crops, and may damage the natural or built environment. Both human activity and natural processes can generate air pollution.

Air pollution is a significant risk factor for a number of pollution-related diseases, including respiratory infections, heart disease, COPD, stroke and lung cancer.[1] The human health effects of poor air quality are far reaching, but principally affect the body's respiratory system and the cardiovascular system. Individual reactions to air pollutants depend on the type of pollutant a person is exposed to, the degree of exposure, and the individual's health status and genetics.[2] Indoor air pollution and poor urban air quality are listed as two of the world's worst toxic pollution problems in the 2008 Blacksmith Institute World's Worst Polluted Places report.[3] Outdoor air pollution alone causes 2.1[4][5] to 4.21 million deaths annually.[1][6] Overall, air pollution causes the deaths of around 7 million people worldwide each year, and is the world's largest single environmental health risk.[1][7][8]

Productivity losses and degraded quality of life caused by air pollution are estimated to cost the world economy $5 trillion per year.[9][10][11] Various pollution control technologies and strategies are available to reduce air pollution.[12][13]

Part of a series on
Weather
Global tropical cyclone tracks-edit2.jpg
Temperate and polar seasons[show]
Tropical seasons[show]
Storms[show]
Precipitation[show]
Topics[show]
Glossaries[show]
Cumulus clouds in fair weather.jpeg Weather portal
vte

Contents
1	Pollutants
1.1	Sources
1.1.1	Anthropogenic (human-made) sources
1.1.2	Natural sources
1.2	Emission factors
2	Exposure
3	Indoor air quality
4	Health effects
4.1	Mortality
4.2	Cardiovascular disease
4.3	Lung disease
4.4	Cancer (lung cancer)
4.5	Children
4.5.1	Infants
4.6	"Clean" areas
4.7	Central nervous system
5	Agricultural effects
6	Economic effects
7	Historical disasters
8	Alternatives to pollution
9	Reduction efforts
9.1	Control devices
10	Regulations
10.1	Canada
10.2	Germany
11	Hotspots
12	Cities
13	Governing urban air pollution
14	Projections
15	See also
16	References
17	Further reading
18	External links
Pollutants
Main articles: Pollutant and Greenhouse gas
An air pollutant is a material in the air that can have adverse effects on humans and the ecosystem. The substance can be solid particles, liquid droplets, or gases. A pollutant can be of natural origin or man-made. Pollutants are classified as primary or secondary. Primary pollutants are usually produced by processes such as ash from a volcanic eruption. Other examples include carbon monoxide gas from motor vehicle exhausts or sulfur dioxide released from factories. Secondary pollutants are not emitted directly. Rather, they form in the air when primary pollutants react or interact. Ground level ozone is a prominent example of a secondary pollutant. Some pollutants may be both primary and secondary: they are both emitted directly and formed from other primary pollutants.


Before flue-gas desulfurization was installed, the emissions from this power plant in New Mexico contained excessive amounts of sulfur dioxide.

Schematic drawing, causes and effects of air pollution: (1) greenhouse effect, (2) particulate contamination, (3) increased UV radiation, (4) acid rain, (5) increased ground-level ozone concentration, (6) increased levels of nitrogen oxides.

Thermal oxidisers are air pollution abatement options for hazardous air pollutants (HAPs), volatile organic compounds (VOCs), and odorous emissions
Pollutants emitted into the atmosphere by human activity include:

Carbon dioxide (CO
2) – Because of its role as a greenhouse gas it has been described as "the leading pollutant"[14] and "the worst climate pollutant".[15] Carbon dioxide is a natural component of the atmosphere, essential for plant life and given off by the human respiratory system.[16] This question of terminology has practical effects, for example as determining whether the U.S. Clean Air Act is deemed to regulate CO
2 emissions.[17] CO
2 currently forms about 410 parts per million (ppm) of earth's atmosphere, compared to about 280 ppm in pre-industrial times,[18] and billions of metric tons of CO
2 are emitted annually by burning of fossil fuels.[19] CO
2 increase in earth's atmosphere has been accelerating.[20]
Sulfur oxides (SOx) – particularly sulfur dioxide, a chemical compound with the formula SO2. SO2 is produced by volcanoes and in various industrial processes. Coal and petroleum often contain sulfur compounds, and their combustion generates sulfur dioxide. Further oxidation of SO2, usually in the presence of a catalyst such as NO2, forms H2SO4, and thus acid rain is formed.[2] This is one of the causes for concern over the environmental impact of the use of these fuels as power sources.
Nitrogen oxides (NOx) – Nitrogen oxides, particularly nitrogen dioxide, are expelled from high temperature combustion, and are also produced during thunderstorms by electric discharge. They can be seen as a brown haze dome above or a plume downwind of cities. Nitrogen dioxide is a chemical compound with the formula NO2. It is one of several nitrogen oxides. One of the most prominent air pollutants, this reddish-brown toxic gas has a characteristic sharp, biting odor.
Carbon monoxide (CO) – CO is a colorless, odorless, toxic gas.[21] It is a product of combustion of fuel such as natural gas, coal or wood. Vehicular exhaust contributes to the majority of carbon monoxide let into our atmosphere. It creates a smog type formation in the air that has been linked to many lung diseases and disruptions to the natural environment and animals.
Volatile organic compounds (VOC) – VOCs are a well-known outdoor air pollutant. They are categorized as either methane (CH4) or non-methane (NMVOCs). Methane is an extremely efficient greenhouse gas which contributes to enhanced global warming. Other hydrocarbon VOCs are also significant greenhouse gases because of their role in creating ozone and prolonging the life of methane in the atmosphere. This effect varies depending on local air quality. The aromatic NMVOCs benzene, toluene and xylene are suspected carcinogens and may lead to leukemia with prolonged exposure. 1,3-butadiene is another dangerous compound often associated with industrial use.
Particulate matter / particles, alternatively referred to as particulate matter (PM), atmospheric particulate matter, or fine particles, are tiny particles of solid or liquid suspended in a gas. In contrast, aerosol refers to combined particles and gas. Some particulates occur naturally, originating from volcanoes, dust storms, forest and grassland fires, living vegetation, and sea spray. Human activities, such as the burning of fossil fuels in vehicles, power plants and various industrial processes also generate significant amounts of aerosols. Averaged worldwide, anthropogenic aerosols—those made by human activities—currently account for approximately 10 percent of our atmosphere. Increased levels of fine particles in the air are linked to health hazards such as heart disease,[22] altered lung function and lung cancer. Particulates are related to respiratory infections and can be particularly harmful to those already suffering from conditions like asthma.[23]
Persistent free radicals connected to airborne fine particles are linked to cardiopulmonary disease.[24][25]
Toxic metals, such as lead and mercury, especially their compounds.
Chlorofluorocarbons (CFCs) – harmful to the ozone layer; emitted from products are currently banned from use. These are gases which are released from air conditioners, refrigerators, aerosol sprays, etc. On release into the air, CFCs rise to the stratosphere. Here they come in contact with other gases and damage the ozone layer. This allows harmful ultraviolet rays to reach the earth's surface. This can lead to skin cancer, eye disease and can even cause damage to plants.
Ammonia – emitted mainly by agricultural waste. Ammonia is a compound with the formula NH3. It is normally encountered as a gas with a characteristic pungent odor. Ammonia contributes significantly to the nutritional needs of terrestrial organisms by serving as a precursor to foodstuffs and fertilizers. Ammonia, either directly or indirectly, is also a building block for the synthesis of many pharmaceuticals. Although in wide use, ammonia is both caustic and hazardous. In the atmosphere, ammonia reacts with oxides of nitrogen and sulfur to form secondary particles.[26]
Odors — such as from garbage, sewage, and industrial processes
Radioactive pollutants – produced by nuclear explosions, nuclear events, war explosives, and natural processes such as the radioactive decay of radon.
Secondary pollutants include:

Particulates created from gaseous primary pollutants and compounds in photochemical smog. Smog is a kind of air pollution. Classic smog results from large amounts of coal burning in an area caused by a mixture of smoke and sulfur dioxide. Modern smog does not usually come from coal but from vehicular and industrial emissions that are acted on in the atmosphere by ultraviolet light from the sun to form secondary pollutants that also combine with the primary emissions to form photochemical smog.
Ground level ozone (O3) formed from NOx and VOCs. Ozone (O3) is a key constituent of the troposphere. It is also an important constituent of certain regions of the stratosphere commonly known as the Ozone layer. Photochemical and chemical reactions involving it drive many of the chemical processes that occur in the atmosphere by day and by night. At abnormally high concentrations brought about by human activities (largely the combustion of fossil fuel), it is a pollutant and a constituent of smog.
Peroxyacetyl nitrate (C2H3NO5) – similarly formed from NOx and VOCs.
Minor air pollutants include:

A large number of minor hazardous air pollutants. Some of these are regulated in USA under the Clean Air Act and in Europe under the Air Framework Directive
A variety of persistent organic pollutants, which can attach to particulates
File:NASA - Human Fingerprint on Global Air Quality.webm
This video provides an overview of a NASA study on the human fingerprint on global air quality.
Persistent organic pollutants (POPs) are organic compounds that are resistant to environmental degradation through chemical, biological, and photolytic processes. Because of this, they have been observed to persist in the environment, to be capable of long-range transport, bioaccumulate in human and animal tissue, biomagnify in food chains, and to have potentially significant impacts on human health and the environment.

Sources
Mean acidifying emissions (air pollution) of different foods per 100g of protein[27]
Food Types	Acidifying Emissions (g SO2eq per 100g protein)
Beef	
343.6
Cheese	
165.5
Pork	
142.7
Lamb and Mutton	
139.0
Farmed Crustaceans	
133.1
Poultry	
102.4
Farmed Fish	
65.9
Eggs	
53.7
Groundnuts	
22.6
Peas	
8.5
Tofu	
6.7
Anthropogenic (human-made) sources

Controlled burning of a field outside of Statesboro, Georgia in preparation for spring planting.

Smoking of fish over an open fire in Ghana, 2018
These are mostly related to the burning of fuel.

Stationary sources include smoke stacks of fossil fuel power stations (see for example environmental impact of the coal industry), manufacturing facilities (factories) and waste incinerators, as well as furnaces and other types of fuel-burning heating devices. In developing and poor countries, traditional biomass burning is the major source of air pollutants; traditional biomass includes wood, crop waste and dung.[28][29]
Mobile sources include motor vehicles, marine vessels, and aircraft.
Controlled burn practices in agriculture and forest management. Controlled or prescribed burning is a technique sometimes used in forest management, farming, prairie restoration or greenhouse gas abatement. Fire is a natural part of both forest and grassland ecology and controlled fire can be a tool for foresters. Controlled burning stimulates the germination of some desirable forest trees, thus renewing the forest.
There are also sources from processes other than combustion

Fumes from paint, hair spray, varnish, aerosol sprays and other solvents. These can be substantial; emissions from these sources was estimated to account for almost half of pollution from volatile organic compounds in the Los Angeles basin in the 2010s.[30]
Waste deposition in landfills, which generate methane. Methane is highly flammable and may form explosive mixtures with air. Methane is also an asphyxiant and may displace oxygen in an enclosed space. Asphyxia or suffocation may result if the oxygen concentration is reduced to below 19.5% by displacement.
Military resources, such as nuclear weapons, toxic gases, germ warfare and rocketry.
Fertilized farmland may be a major source of nitrogen oxides.[31]
Natural sources

Dust storm approaching Stratford, Texas.
Dust from natural sources, usually large areas of land with little vegetation or no vegetation
Methane, emitted by the digestion of food by animals, for example cattle
Radon gas from radioactive decay within the Earth's crust. Radon is a colorless, odorless, naturally occurring, radioactive noble gas that is formed from the decay of radium. It is considered to be a health hazard. Radon gas from natural sources can accumulate in buildings, especially in confined areas such as the basement and it is the second most frequent cause of lung cancer, after cigarette smoking.
Smoke and carbon monoxide from wildfires. During periods of actives wildfires, smoke from uncontrolled biomass combustion can make up almost 75% of all air pollution by concentration.[32]
Vegetation, in some regions, emits environmentally significant amounts of Volatile organic compounds (VOCs) on warmer days. These VOCs react with primary anthropogenic pollutants—specifically, NOx, SO2, and anthropogenic organic carbon compounds — to produce a seasonal haze of secondary pollutants.[33] Black gum, poplar, oak and willow are some examples of vegetation that can produce abundant VOCs. The VOC production from these species result in ozone levels up to eight times higher than the low-impact tree species.[34]
Volcanic activity, which produces sulfur, chlorine, and ash particulates
Emission factors
Main article: AP 42 Compilation of Air Pollutant Emission Factors

Beijing air on a 2005-day after rain (left) and a smoggy day (right)
Air pollutant emission factors are reported representative values that attempt to relate the quantity of a pollutant released to the ambient air with an activity associated with the release of that pollutant. These factors are usually expressed as the weight of pollutant divided by a unit weight, volume, distance, or duration of the activity emitting the pollutant (e.g., kilograms of particulate emitted per tonne of coal burned). Such factors facilitate estimation of emissions from various sources of air pollution. In most cases, these factors are simply averages of all available data of acceptable quality, and are generally assumed to be representative of long-term averages.

There are 12 compounds in the list of persistent organic pollutants. Dioxins and furans are two of them and intentionally created by combustion of organics, like open burning of plastics. These compounds are also endocrine disruptors and can mutate the human genes.

The United States Environmental Protection Agency has published a compilation of air pollutant emission factors for a wide range of industrial sources.[35] The United Kingdom, Australia, Canada and many other countries have published similar compilations, as well as the European Environment Agency.[36][37][38][39]

Exposure

Up to 30 % of Europeans living in cities are exposed to air pollutant levels exceeding EU air quality standards. And around 98 % of Europeans living in cities are exposed to levels of air pollutants deemed damaging to health by the World Health Organization's more stringent guidelines.[40]
Air pollution risk is a function of the hazard of the pollutant and the exposure to that pollutant. Air pollution exposure can be expressed for an individual, for certain groups (e.g. neighborhoods or children living in a country), or for entire populations. For example, one may want to calculate the exposure to a hazardous air pollutant for a geographic area, which includes the various microenvironments and age groups. This can be calculated[2] as an inhalation exposure. This would account for daily exposure in various settings (e.g. different indoor micro-environments and outdoor locations). The exposure needs to include different age and other demographic groups, especially infants, children, pregnant women and other sensitive subpopulations. The exposure to an air pollutant must integrate the concentrations of the air pollutant with respect to the time spent in each setting and the respective inhalation rates for each subgroup for each specific time that the subgroup is in the setting and engaged in particular activities (playing, cooking, reading, working, spending time in traffic, etc.). For example, a small child's inhalation rate will be less than that of an adult. A child engaged in vigorous exercise will have a higher respiration rate than the same child in a sedentary activity. The daily exposure, then, needs to reflect the time spent in each micro-environmental setting and the type of activities in these settings. The air pollutant concentration in each microactivity/microenvironmental setting is summed to indicate the exposure.[2] For some pollutants such as black carbon, traffic related exposures may dominate total exposure despite short exposure times since high concentrations coincide with proximity to major roads or participation to (motorized) traffic.[41] A large portion of total daily exposure occurs as short peaks of high concentrations, but it remains unclear how to define peaks and determine their frequency and health impact.[42]

Indoor air quality
Main article: Indoor air quality

Air quality monitoring, New Delhi, India.
A lack of ventilation indoors concentrates air pollution where people often spend the majority of their time. Radon (Rn) gas, a carcinogen, is exuded from the Earth in certain locations and trapped inside houses. Building materials including carpeting and plywood emit formaldehyde (H2CO) gas. Paint and solvents give off volatile organic compounds (VOCs) as they dry. Lead paint can degenerate into dust and be inhaled. Intentional air pollution is introduced with the use of air fresheners, incense, and other scented items. Controlled wood fires in cook stoves and fireplaces can add significant amounts of harmful smoke particulates into the air, inside and out.[43][44] Indoor pollution fatalities may be caused by using pesticides and other chemical sprays indoors without proper ventilation.

Carbon monoxide poisoning and fatalities are often caused by faulty vents and chimneys, or by the burning of charcoal indoors or in a confined space, such as a tent.[45] Chronic carbon monoxide poisoning can result even from poorly-adjusted pilot lights. Traps are built into all domestic plumbing to keep sewer gas and hydrogen sulfide, out of interiors. Clothing emits tetrachloroethylene, or other dry cleaning fluids, for days after dry cleaning.

Though its use has now been banned in many countries, the extensive use of asbestos in industrial and domestic environments in the past has left a potentially very dangerous material in many localities. Asbestosis is a chronic inflammatory medical condition affecting the tissue of the lungs. It occurs after long-term, heavy exposure to asbestos from asbestos-containing materials in structures. Sufferers have severe dyspnea (shortness of breath) and are at an increased risk regarding several different types of lung cancer. As clear explanations are not always stressed in non-technical literature, care should be taken to distinguish between several forms of relevant diseases. According to the World Health Organization (WHO), these may defined as; asbestosis, lung cancer, and Peritoneal Mesothelioma (generally a very rare form of cancer, when more widespread it is almost always associated with prolonged exposure to asbestos).

Biological sources of air pollution are also found indoors, as gases and airborne particulates. Pets produce dander, people produce dust from minute skin flakes and decomposed hair, dust mites in bedding, carpeting and furniture produce enzymes and micrometre-sized fecal droppings, inhabitants emit methane, mold forms on walls and generates mycotoxins and spores, air conditioning systems can incubate Legionnaires' disease and mold, and houseplants, soil and surrounding gardens can produce pollen, dust, and mold. Indoors, the lack of air circulation allows these airborne pollutants to accumulate more than they would otherwise occur in nature.

Health effects
See also: Neuroplastic effects of pollution
In 2012, air pollution caused premature deaths on average of 1 year in Europe, and was a significant risk factor for a number of pollution-related diseases, including respiratory infections, heart disease, COPD, stroke and lung cancer.[1] The health effects caused by air pollution may include difficulty in breathing, wheezing, coughing, asthma and worsening of existing respiratory and cardiac conditions. These effects can result in increased medication use, increased doctor or emergency department visits, more hospital admissions and premature death. The human health effects of poor air quality are far reaching, but principally affect the body's respiratory system and the cardiovascular system. Individual reactions to air pollutants depend on the type of pollutant a person is exposed to, the degree of exposure, and the individual's health status and genetics.[2] The most common sources of air pollution include particulates, ozone, nitrogen dioxide, and sulfur dioxide. Children aged less than five years that live in developing countries are the most vulnerable population in terms of total deaths attributable to indoor and outdoor air pollution.[46]

Mortality

Absolute number of deaths from ambient particulate air pollution[47]
The World Health Organization estimated in 2014 that every year air pollution causes the premature death of some 7 million people worldwide.[1] Studies published in March 2019 indicated that the number may be around 8.8 million.[48]

India has the highest death rate due to air pollution.[49] India also has more deaths from asthma than any other nation according to the World Health Organization. In December 2013 air pollution was estimated to kill 500,000 people in China each year.[50] There is a positive correlation between pneumonia-related deaths and air pollution from motor vehicle emissions.[51]

Annual premature European deaths caused by air pollution are estimated at 430,000[52]-800,000[48] An important cause of these deaths is nitrogen dioxide and other nitrogen oxides (NOx) emitted by road vehicles.[52] In a 2015 consultation document the UK government disclosed that nitrogen dioxide is responsible for 23,500 premature UK deaths per annum.[53] Across the European Union, air pollution is estimated to reduce life expectancy by almost nine months.[54] Causes of deaths include strokes, heart disease, COPD, lung cancer, and lung infections.[1]

Urban outdoor air pollution is estimated to cause 1.3 million deaths worldwide per year. Children are particularly at risk due to the immaturity of their respiratory organ systems.[55]

The US EPA estimated in 2004 that a proposed set of changes in diesel engine technology (Tier 2) could result in 12,000 fewer premature mortalities, 15,000 fewer heart attacks, 6,000 fewer emergency department visits by children with asthma, and 8,900 fewer respiratory-related hospital admissions each year in the United States.[56]

The US EPA has estimated that limiting ground-level ozone concentration to 65 parts per billion, would avert 1,700 to 5,100 premature deaths nationwide in 2020 compared with the 75-ppb standard. The agency projected the more protective standard would also prevent an additional 26,000 cases of aggravated asthma, and more than a million cases of missed work or school.[57][58] Following this assessment, the EPA acted to protect public health by lowering the National Ambient Air Quality Standards (NAAQS) for ground-level ozone to 70 parts per billion (ppb).[59]

A new economic study of the health impacts and associated costs of air pollution in the Los Angeles Basin and San Joaquin Valley of Southern California shows that more than 3,800 people die prematurely (approximately 14 years earlier than normal) each year because air pollution levels violate federal standards. The number of annual premature deaths is considerably higher than the fatalities related to auto collisions in the same area, which average fewer than 2,000 per year.[60][61][62]

Diesel exhaust (DE) is a major contributor to combustion-derived particulate matter air pollution. In several human experimental studies, using a well-validated exposure chamber setup, DE has been linked to acute vascular dysfunction and increased thrombus formation.[63][64]

The mechanisms linking air pollution to increased cardiovascular mortality are uncertain, but probably include pulmonary and systemic inflammation.[65]

A study by Greenpeace estimate there are 4.5 million annual premature deaths worldwide because of pollutants released by high-emission power stations and vehicle exhausts, 65,000 deaths occur in the Middle East each year due to pollution.[66]

Cardiovascular disease
A 2007 review of evidence found that, ambient air pollution exposure is a risk factor correlating with increased total mortality from cardiovascular events (range: 12% to 14% per 10 µg/m3 increase).[67][clarification needed]

Air pollution is also emerging as a risk factor for stroke, particularly in developing countries where pollutant levels are highest.[68] A 2007 study found that in women, air pollution is not associated with hemorrhagic but with ischemic stroke.[69] Air pollution was also found to be associated with increased incidence and mortality from coronary stroke in a cohort study in 2011.[70] Associations are believed to be causal and effects may be mediated by vasoconstriction, low-grade inflammation and atherosclerosis[71] Other mechanisms such as autonomic nervous system imbalance have also been suggested.[72] [73]

Lung disease
Research has demonstrated increased risk of developing asthma[74] and COPD[75] from increased exposure to traffic-related air pollution. Additionally, air pollution has been associated with increased hospitalization and mortality from asthma and COPD.[76][77] Chronic obstructive pulmonary disease (COPD) includes diseases such as chronic bronchitis and emphysema.[78]

A study conducted in 1960–1961 in the wake of the Great Smog of 1952 compared 293 London residents with 477 residents of Gloucester, Peterborough, and Norwich, three towns with low reported death rates from chronic bronchitis. All subjects were male postal truck drivers aged 40 to 59. Compared to the subjects from the outlying towns, the London subjects exhibited more severe respiratory symptoms (including cough, phlegm, and dyspnea), reduced lung function (FEV1 and peak flow rate), and increased sputum production and purulence. The differences were more pronounced for subjects aged 50 to 59. The study controlled for age and smoking habits, so concluded that air pollution was the most likely cause of the observed differences.[79] More recent studies have shown that air pollution exposure from traffic reduces lung function development in children [80] and lung function may be compromised by air pollution even at low concentrations.[81] Air pollution exposure also cause lung cancer in non smokers.

It is believed that much like cystic fibrosis, by living in a more urban environment serious health hazards become more apparent. Studies have shown that in urban areas patients suffer mucus hypersecretion, lower levels of lung function, and more self-diagnosis of chronic bronchitis and emphysema.[82]

Cancer (lung cancer)

Unprotected exposure to PM2.5 air pollution can be equivalent to smoking multiple cigarettes per day,[83] potentially increasing the risk of cancer, which is mainly the result of environmental factors.[84]
A review of evidence regarding whether ambient air pollution exposure is a risk factor for cancer in 2007 found solid data to conclude that long-term exposure to PM2.5 (fine particulates) increases the overall risk of non-accidental mortality by 6% per a 10 microg/m3 increase. Exposure to PM2.5 was also associated with an increased risk of mortality from lung cancer (range: 15% to 21% per 10 microg/m3 increase) and total cardiovascular mortality (range: 12% to 14% per a 10 microg/m3 increase). The review further noted that living close to busy traffic appears to be associated with elevated risks of these three outcomes – increase in lung cancer deaths, cardiovascular deaths, and overall non-accidental deaths. The reviewers also found suggestive evidence that exposure to PM2.5 is positively associated with mortality from coronary heart diseases and exposure to SO2 increases mortality from lung cancer, but the data was insufficient to provide solid conclusions.[85] Another investigation showed that higher activity level increases deposition fraction of aerosol particles in human lung and recommended avoiding heavy activities like running in outdoor space at polluted areas.[86]

In 2011, a large Danish epidemiological study found an increased risk of lung cancer for patients who lived in areas with high nitrogen oxide concentrations. In this study, the association was higher for non-smokers than smokers.[87] An additional Danish study, also in 2011, likewise noted evidence of possible associations between air pollution and other forms of cancer, including cervical cancer and brain cancer.[88]

Children
In the United States, despite the passage of the Clean Air Act in 1970, in 2002 at least 146 million Americans were living in non-attainment areas—regions in which the concentration of certain air pollutants exceeded federal standards.[89] These dangerous pollutants are known as the criteria pollutants, and include ozone, particulate matter, sulfur dioxide, nitrogen dioxide, carbon monoxide, and lead. Protective measures to ensure children's health are being taken in cities such as New Delhi, India where buses now use compressed natural gas to help eliminate the "pea-soup" smog.[90] A recent study in Europe has found that exposure to ultrafine particles can increase blood pressure in children.[91] According to a WHO report-2018, polluted air is a main cause poisoning millions of children under the age of 15 years and ruining their lives which resulting to death of some six hundred thousand children annually.[92]

Infants
Ambient levels of air pollution have been associated with preterm birth and low birth weight. A 2014 WHO worldwide survey on maternal and perinatal health found a statistically significant association between low birth weights (LBW) and increased levels of exposure to PM2.5. Women in regions with greater than average PM2.5 levels had statistically significant higher odds of pregnancy resulting in a low-birth weight infant even when adjusted for country-related variables.[93] The effect is thought to be from stimulating inflammation and increasing oxidative stress.

A study by the University of York found that in 2010 exposure to PM2.5 was strongly associated with 18% of preterm births globally, which was approximately 2.7 million premature births. The countries with the highest air pollution associated preterm births were in South and East Asia, the Middle East, North Africa, and West sub-Saharan Africa.[94]

The source of PM 2.5 differs greatly by region. In South and East Asia, pregnant women are frequently exposed to indoor air pollution because of the wood and other biomass fuels used for cooking which are responsible for more than 80% of regional pollution. In the Middle East, North Africa and West sub-Saharan Africa, fine PM comes from natural sources, such as dust storms.[94] The United States had an estimated 50,000 preterm births associated with exposure to PM2.5 in 2010.[94]

A study performed by Wang, et al. between the years of 1988 and 1991 has found a correlation between sulfur Dioxide (SO2) and total suspended particulates (TSP) and preterm births and low birth weights in Beijing. A group of 74,671 pregnant women, in four separate regions of Beijing, were monitored from early pregnancy to delivery along with daily air pollution levels of sulfur Dioxide and TSP (along with other particulates). The estimated reduction in birth weight was 7.3 g for every 100 µg/m3 increase in SO2 and 6.9 g for each 100 µg/m3 increase in TSP. These associations were statistically significant in both summer and winter, although, summer was greater. The proportion of low birth weight attributable to air pollution, was 13%. This is the largest attributable risk ever reported for the known risk factors of low birth weight.[95] Coal stoves, which are in 97% of homes, are a major source of air pollution in this area.

Brauer et al. studied the relationship between air pollution and proximity to a highway with pregnancy outcomes in a Vancouver cohort of pregnant woman using addresses to estimate exposure during pregnancy. Exposure to NO, NO2, CO PM10 and PM2.5 were associated with infants born small for gestational age (SGA). Women living less than 50 meters away from an expressway or highway were 26% more likely to give birth to a SGA infant.[96]

"Clean" areas
Even in the areas with relatively low levels of air pollution, public health effects can be significant and costly, since a large number of people breathe in such pollutants. A study published in 2017 found that even in areas of the U.S. where ozone and PM2.5 meet federal standards, Medicare recipients who are exposed to more air pollution have higher mortality rates.[97] A 2005 scientific study for the British Columbia Lung Association showed that a small improvement in air quality (1% reduction of ambient PM2.5 and ozone concentrations) would produce $29 million in annual savings in the Metro Vancouver region in 2010.[98] This finding is based on health valuation of lethal (death) and sub-lethal (illness) affects.

In 2020, scientists found that the boundary layer air over the Southern Ocean around Antarctica is unpolluted by humans.[99]

Central nervous system
Data is accumulating that air pollution exposure also affects the central nervous system.[100]

In a June 2014 study conducted by researchers at the University of Rochester Medical Center, published in the journal Environmental Health Perspectives, it was discovered that early exposure to air pollution causes the same damaging changes in the brain as autism and schizophrenia. The study also shows that air pollution also affected short-term memory, learning ability, and impulsivity. Lead researcher Professor Deborah Cory-Slechta said that "When we looked closely at the ventricles, we could see that the white matter that normally surrounds them hadn't fully developed. It appears that inflammation had damaged those brain cells and prevented that region of the brain from developing, and the ventricles simply expanded to fill the space. Our findings add to the growing body of evidence that air pollution may play a role in autism, as well as in other neurodevelopmental disorders." In a study of mice, air pollution also has a more significant negative effect on males than on females.[101][102][103]

In 2015, experimental studies reported the detection of significant episodic (situational) cognitive impairment from impurities in indoor air breathed by test subjects who were not informed about changes in the air quality. Researchers at the Harvard University and SUNY Upstate Medical University and Syracuse University measured the cognitive performance of 24 participants in three different controlled laboratory atmospheres that simulated those found in "conventional" and "green" buildings, as well as green buildings with enhanced ventilation. Performance was evaluated objectively using the widely used Strategic Management Simulation software simulation tool, which is a well-validated assessment test for executive decision-making in an unconstrained situation allowing initiative and improvisation. Significant deficits were observed in the performance scores achieved in increasing concentrations of either volatile organic compounds (VOCs) or carbon dioxide, while keeping other factors constant. The highest impurity levels reached are not uncommon in some classroom or office environments.[104][105] Air pollution increases the risk of dementia in people over 50 years old.[106]

Agricultural effects
In India in 2014, it was reported that air pollution by black carbon and ground level ozone had reduced crop yields in the most affected areas by almost half in 2011 when compared to 1980 levels.[107]

Economic effects
Air pollution costs the world economy $5 trillion per year as a result of productivity losses and degraded quality of life, according to a joint study by the World Bank and the Institute for Health Metrics and Evaluation (IHME) at the University of Washington.[9][10][11] These productivity losses are caused by deaths due to diseases caused by air pollution. One out of ten deaths in 2013 was caused by diseases associated with air pollution and the problem is getting worse. The problem is even more acute in the developing world. "Children under age 5 in lower-income countries are more than 60 times as likely to die from exposure to air pollution as children in high-income countries."[9][10] The report states that additional economic losses caused by air pollution, including health costs[108] and the adverse effect on agricultural and other productivity were not calculated in the report, and thus the actual costs to the world economy are far higher than $5 trillion.

Historical disasters
The world's worst short-term civilian pollution crisis was the 1984 Bhopal Disaster in India.[109] Leaked industrial vapours from the Union Carbide factory, belonging to Union Carbide, Inc., U.S.A. (later bought by Dow Chemical Company), killed at least 3787 people and injured from 150,000 to 600,000. The United Kingdom suffered its worst air pollution event when the December 4 Great Smog of 1952 formed over London. In six days more than 4,000 died and more recent estimates put the figure at nearer 12,000.[110] An accidental leak of anthrax spores from a biological warfare laboratory in the former USSR in 1979 near Sverdlovsk is believed to have caused at least 64 deaths.[111] The worst single incident of air pollution to occur in the US occurred in Donora, Pennsylvania in late October, 1948, when 20 people died and over 7,000 were injured.[112]

Alternatives to pollution
There are now practical alternatives to the principal causes of air pollution:

Areas downwind (over 20 miles) of major airports more than double total particulate emissions in air, even when factoring in areas with frequent ship calls, and heavy freeway and city traffic like Los Angeles.[113] Aviation biofuel mixed in with jetfuel at a 50/50 ratio can reduce jet derived cruise altitude particulate emissions by 50–70%, according to a NASA led 2017 study (however, this should imply ground level benefits to urban air pollution as well).[114]
Ship propulsion and idling can be switched to much cleaner fuels like natural gas. (Ideally a renewable source but not practical yet)
Combustion of fossil fuels for space heating can be replaced by using ground source heat pumps and seasonal thermal energy storage.[115]
Electric power generation from burning fossil fuels can be replaced by power generation from nuclear and renewables. For poor nations, heating and home stoves that contribute much to regional air pollution can be replaced by a much cleaner fossil fuel like natural gas, or ideally, renewables.
Motor vehicles driven by fossil fuels, a key factor in urban air pollution, can be replaced by electric vehicles. Though lithium supply and cost is a limitation, there are alternatives. Herding more people into clean public transit such as electric trains can also help. Nevertheless, even in emission-free electric vehicles, rubber tires produce significant amounts of air pollution themselves, ranking as 13th worst pollutant in Los Angeles.[116]
Reducing travel in vehicles can curb pollution. After Stockholm reduced vehicle traffic in the central city with a congestion tax, nitrogen dioxide and PM10 pollution declined, as did acute pediatric asthma attacks.[117]
Biodigesters can be utilized in poor nations where slash and burn is prevalent, turning a useless commodity into a source of income. The plants can be gathered and sold to a central authority that will break it down in a large modern biodigester, producing much needed energy to use.
Induced humidity and ventilation both can greatly dampen air pollution in enclosed spaces, which was found to be relatively high inside subway lines due to braking and friction and relatively less ironically inside transit buses than lower sitting passenger automobiles or subways.[118]
Reduction efforts
Various pollution control technologies and strategies are available to reduce air pollution.[12][13] At its most basic level, land-use planning is likely to involve zoning and transport infrastructure planning. In most developed countries, land-use planning is an important part of social policy, ensuring that land is used efficiently for the benefit of the wider economy and population, as well as to protect the environment.

Because a large share of air pollution is caused by combustion of fossil fuels such as coal and oil, the reduction of these fuels can reduce air pollution drastically. Most effective is the switch to clean power sources such as wind power, solar power, hydro power which don't cause air pollution.[119] Efforts to reduce pollution from mobile sources includes primary regulation (many developing countries have permissive regulations),[citation needed] expanding regulation to new sources (such as cruise and transport ships, farm equipment, and small gas-powered equipment such as string trimmers, chainsaws, and snowmobiles), increased fuel efficiency (such as through the use of hybrid vehicles), conversion to cleaner fuels or conversion to electric vehicles.

Titanium dioxide has been researched for its ability to reduce air pollution. Ultraviolet light will release free electrons from material, thereby creating free radicals, which break up VOCs and NOx gases. One form is superhydrophilic.[120]

In 2014, Prof. Tony Ryan and Prof. Simon Armitage of University of Sheffield prepared a 10 meter by 20 meter-sized poster coated with microscopic, pollution-eating nanoparticles of titanium dioxide. Placed on a building, this giant poster can absorb the toxic emission from around 20 cars each day.[121]

A very effective means to reduce air pollution is the transition to renewable energy. According to a study published in Energy and Environmental Science in 2015 the switch to 100% renewable energy in the United States would eliminate about 62,000 premature mortalities per year and about 42,000 in 2050, if no biomass were used. This would save about $600 billion in health costs a year due to reduced air pollution in 2050, or about 3.6% of the 2014 U.S. gross domestic product.[119]

There is limited evidence that efforts to reduce particulate matter in the air can result in better health in Africa, the Middle East, Eastern Europe, Central Asia, and Southeast Asia.[122]

Control devices
The following items are commonly used as pollution control devices in industry and transportation. They can either destroy contaminants or remove them from an exhaust stream before it is emitted into the atmosphere.

Particulate control
Mechanical collectors (dust cyclones, multicyclones)
Electrostatic precipitators An electrostatic precipitator (ESP), or electrostatic air cleaner is a particulate collection device that removes particles from a flowing gas (such as air), using the force of an induced electrostatic charge. Electrostatic precipitators are highly efficient filtration devices that minimally impede the flow of gases through the device, and can easily remove fine particulates such as dust and smoke from the air stream.
Baghouses Designed to handle heavy dust loads, a dust collector consists of a blower, dust filter, a filter-cleaning system, and a dust receptacle or dust removal system (distinguished from air cleaners which utilize disposable filters to remove the dust).
Particulate scrubbers Wet scrubber is a form of pollution control technology. The term describes a variety of devices that use pollutants from a furnace flue gas or from other gas streams. In a wet scrubber, the polluted gas stream is brought into contact with the scrubbing liquid, by spraying it with the liquid, by forcing it through a pool of liquid, or by some other contact method, so as to remove the pollutants.
Scrubbers
Baffle spray scrubber
Cyclonic spray scrubber
Ejector venturi scrubber
Mechanically aided scrubber
Spray tower
Wet scrubber
NOx control
Low NOx burners
Selective catalytic reduction (SCR)
Selective non-catalytic reduction (SNCR)
NOx scrubbers
Exhaust gas recirculation
Catalytic converter (also for VOC control)
VOC abatement
Adsorption systems, using activated carbon, such as Fluidized Bed Concentrator
Flares
Thermal oxidizers
Catalytic converters
Biofilters
Absorption (scrubbing)
Cryogenic condensers
Vapor recovery systems
Acid Gas/SO2 control
Wet scrubbers
Dry scrubbers
Flue-gas desulfurization
Mercury control
Sorbent Injection Technology
Electro-Catalytic Oxidation (ECO)
K-Fuel
Dioxin and furan control
Miscellaneous associated equipment
Source capturing systems
Continuous emissions monitoring systems (CEMS)
Regulations

Smog in Cairo
Main article: Air quality law
In general, there are two types of air quality standards. The first class of standards (such as the U.S. National Ambient Air Quality Standards and E.U. Air Quality Directive) set maximum atmospheric concentrations for specific pollutants. Environmental agencies enact regulations which are intended to result in attainment of these target levels. The second class (such as the North American air quality index) take the form of a scale with various thresholds, which is used to communicate to the public the relative risk of outdoor activity. The scale may or may not distinguish between different pollutants.

Canada
In Canada, air pollution and associated health risks are measured with the Air Quality Health Index or (AQHI). It is a health protection tool used to make decisions to reduce short-term exposure to air pollution by adjusting activity levels during increased levels of air pollution.

The Air Quality Health Index or "AQHI" is a federal program jointly coordinated by Health Canada and Environment Canada. However, the AQHI program would not be possible without the commitment and support of the provinces, municipalities and NGOs. From air quality monitoring to health risk communication and community engagement, local partners are responsible for the vast majority of work related to AQHI implementation. The AQHI provides a number from 1 to 10+ to indicate the level of health risk associated with local air quality. Occasionally, when the amount of air pollution is abnormally high, the number may exceed 10. The AQHI provides a local air quality current value as well as a local air quality maximums forecast for today, tonight and tomorrow and provides associated health advice.

1	2	3	4	5	6	7	8	9	10	+
Risk:	Low (1-–3)	Moderate (4-–6)	High (7-–10)	Very high (above 10)
As it is now known that even low levels of air pollution can trigger discomfort for the sensitive population, the index has been developed as a continuum: The higher the number, the greater the health risk and need to take precautions. The index describes the level of health risk associated with this number as 'low', 'moderate', 'high' or 'very high', and suggests steps that can be taken to reduce exposure.[123]

Health Risk	Air Quality Health Index	Health Messages[124]
At Risk population	General Population
Low	'-1–3'	Enjoy your usual outdoor activities.	Ideal air quality for outdoor activities
Moderate	'-4–6'	Consider reducing or rescheduling strenuous activities outdoors if you are experiencing symptoms.	No need to modify your usual outdoor activities unless you experience symptoms such as coughing and throat irritation.
High	'-7–10'	Reduce or reschedule strenuous activities outdoors. Children and the elderly should also take it easy.	Consider reducing or rescheduling strenuous activities outdoors if you experience symptoms such as coughing and throat irritation.
Very high	Above 10	Avoid strenuous activities outdoors. Children and the elderly should also avoid outdoor physical exertion and should stay indoors.	Reduce or reschedule strenuous activities outdoors, especially if you experience symptoms such as coughing and throat irritation.
The measurement is based on the observed relationship of Nitrogen Dioxide (NO2), ground-level Ozone (O3) and particulates (PM2.5) with mortality, from an analysis of several Canadian cities. Significantly, all three of these pollutants can pose health risks, even at low levels of exposure, especially among those with pre-existing health problems.

When developing the AQHI, Health Canada's original analysis of health effects included five major air pollutants: particulates, ozone, and nitrogen dioxide (NO2), as well as sulfur dioxide (SO2), and carbon monoxide (CO). The latter two pollutants provided little information in predicting health effects and were removed from the AQHI formulation.

The AQHI does not measure the effects of odour, pollen, dust, heat or humidity.

Germany
TA Luft is the German air quality regulation.

Hotspots
Main article: Toxic Hotspots
Air pollution hotspots are areas where air pollution emissions expose individuals to increased negative health effects.[125] They are particularly common in highly populated, urban areas, where there may be a combination of stationary sources (e.g. industrial facilities) and mobile sources (e.g. cars and trucks) of pollution. Emissions from these sources can cause respiratory disease, childhood asthma, cancer, and other health problems. Fine particulate matter such as diesel soot, which contributes to more than 3.2 million premature deaths around the world each year, is a significant problem. It is very small and can lodge itself within the lungs and enter the bloodstream. Diesel soot is concentrated in densely populated areas, and one in six people in the U.S. live near a diesel pollution hot spot.[126]

External video
video icon AirVisual Earth – realtime map of global wind and air pollution [127]
While air pollution hotspots affect a variety of populations, some groups are more likely to be located in hotspots. Previous studies have shown disparities in exposure to pollution by race and/or income. Hazardous land uses (toxic storage and disposal facilities, manufacturing facilities, major roadways) tend to be located where property values and income levels are low. Low socioeconomic status can be a proxy for other kinds of social vulnerability, including race, a lack of ability to influence regulation and a lack of ability to move to neighborhoods with less environmental pollution. These communities bear a disproportionate burden of environmental pollution and are more likely to face health risks such as cancer or asthma.[128]

Studies show that patterns in race and income disparities not only indicate a higher exposure to pollution but also higher risk of adverse health outcomes.[129] Communities characterized by low socioeconomic status and racial minorities can be more vulnerable to cumulative adverse health impacts resulting from elevated exposure to pollutants than more privileged communities.[129] Blacks and Latinos generally face more pollution than whites and Asians, and low-income communities bear a higher burden of risk than affluent ones.[128] Racial discrepancies are particularly distinct in suburban areas of the Southern United States and metropolitan areas of the Midwestern and Western United States.[130] Residents in public housing, who are generally low-income and cannot move to healthier neighborhoods, are highly affected by nearby refineries and chemical plants.[131]

Cities
See also: List of most polluted cities in the world by particulate matter concentration

Nitrogen dioxide concentrations as measured from satellite 2002–2004

Deaths from air pollution in 2004
Air pollution is usually concentrated in densely populated metropolitan areas, especially in developing countries where environmental regulations are relatively lax or nonexistent.[132] However, even populated areas in developed countries attain unhealthy levels of pollution, with Los Angeles and Rome being two examples.[133] Between 2002 and 2011 the incidence of lung cancer in Beijing near doubled. While smoking remains the leading cause of lung cancer in China, the number of smokers is falling while lung cancer rates are rising.[134]

Most polluted cities by PM[135]
Particulate
matter,
μg/m3 (2004)	City
168	Cairo, Egypt
150	Delhi, India
128	Kolkata, India (Calcutta)
125	Tianjin, China
123	Chongqing, China
109	Kanpur, India
109	Lucknow, India
104	Jakarta, Indonesia
101	Shenyang, China
Governing urban air pollution
Further information: Phase-out of fossil fuel vehicles § Cities and territories
In Europe, Council Directive 96/62/EC on ambient air quality assessment and management provides a common strategy against which member states can "set objectives for ambient air quality in order to avoid, prevent or reduce harmful effects on human health and the environment ... and improve air quality where it is unsatisfactory".[136]

On 25 July 2008 in the case Dieter Janecek v Freistaat Bayern CURIA, the European Court of Justice ruled that under this directive[136] citizens have the right to require national authorities to implement a short term action plan that aims to maintain or achieve compliance to air quality limit values.[137]

This important case law appears to confirm the role of the EC as centralised regulator to European nation-states as regards air pollution control. It places a supranational legal obligation on the UK to protect its citizens from dangerous levels of air pollution, furthermore superseding national interests with those of the citizen.

In 2010, the European Commission (EC) threatened the UK with legal action against the successive breaching of PM10 limit values.[138] The UK government has identified that if fines are imposed, they could cost the nation upwards of £300 million per year.[139]

In March 2011, the Greater London Built-up Area remains the only UK region in breach of the EC's limit values, and has been given 3 months to implement an emergency action plan aimed at meeting the EU Air Quality Directive.[140] The City of London has dangerous levels of PM10 concentrations, estimated to cause 3000 deaths per year within the city.[141] As well as the threat of EU fines, in 2010 it was threatened with legal action for scrapping the western congestion charge zone, which is claimed to have led to an increase in air pollution levels.[142]

In response to these charges, Boris Johnson, Mayor of London, has criticised the current need for European cities to communicate with Europe through their nation state's central government, arguing that in future "A great city like London" should be permitted to bypass its government and deal directly with the European Commission regarding its air quality action plan.[140]

This can be interpreted as recognition that cities can transcend the traditional national government organisational hierarchy and develop solutions to air pollution using global governance networks, for example through transnational relations. Transnational relations include but are not exclusive to national governments and intergovernmental organisations,[143] allowing sub-national actors including cities and regions to partake in air pollution control as independent actors.

Particularly promising at present are global city partnerships.[144] These can be built into networks, for example the C40 Cities Climate Leadership Group, of which London is a member. The C40 is a public 'non-state' network of the world's leading cities that aims to curb their greenhouse emissions.[144] The C40 has been identified as 'governance from the middle' and is an alternative to intergovernmental policy.[145] It has the potential to improve urban air quality as participating cities "exchange information, learn from best practices and consequently mitigate carbon dioxide emissions independently from national government decisions".[144] A criticism of the C40 network is that its exclusive nature limits influence to participating cities and risks drawing resources away from less powerful city and regional actors."""



# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['air']

# Most similar words
similar = model.wv.most_similar('pollution')
prompt_mapper = {"α*1E7, K-1： Thermal expansion coefficient temp.range, °C： Temperature range":"thermal expansion coefficient at different temperatures",
                "Glass transition temperature": "Glass transition temperature",
                "Density" : "Density",
                "T, °C： Temperature log(η, P)： η: Viscosity" : "temperature at given visocity",
                 "E: Young's modulus": "Young's modulus",
                 "Thermal expansion coefficient":"Thermal expansion coefficient",
                 "Refractive index" : "Refractive index",
                 "H: Microhardness" : "Hardness",
                 "Crystallization temperature": "Crystallization temperature",
                 "τ, %： Transmittance λ, nm： Wavelength" : "transmittance at different wavelengths",
                 "Liquidus temperature": "Liquidus temperature"         
        }

prompt_mapper_extraction = {'α*1E7, K-1： Thermal expansion coefficient temp.range, °C： Temperature range': 'CTE',
 'Glass transition temperature': 'Tg',
 'Density': 'rho',
 'T, °C： Temperature log(η, P)： η: Viscosity': 'T_η',
 "E: Young's modulus": "E",
 'Thermal expansion coefficient': 'CTE',
 'Refractive index': 'n',
 'H: Microhardness': 'H',
 'Crystallization temperature': 'Tx',
 'τ, %： Transmittance λ, nm： Wavelength': 'τ',
 'Liquidus temperature': 'Tl'}

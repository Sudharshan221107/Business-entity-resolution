import os
import random
import csv

def generate_benchmark_data(base_dir="dataset", seed=42):
    random.seed(seed)
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Base templates of real-world style businesses
    us_businesses = [
        ("Apex Global Logistics", "1200 Industrial Pkwy, Suite 300, Cleveland, OH 44135", "US"),
        ("Blue Horizon Financial Services", "450 Lexington Ave, 14th Floor, New York, NY 10017", "US"),
        ("Cascade Mountain Healthcare Clinic", "8820 Wilshire Blvd, Suite 210, Beverly Hills, CA 90211", "US"),
        ("Delta Precision Manufacturing Inc", "5500 Technology Dr, Building B, Austin, TX 78754", "US"),
        ("Echo River Software Solutions", "101 California St, Suite 1900, San Francisco, CA 94111", "US"),
        ("Frontier Energy Partners", "1000 Louisiana St, Suite 4300, Houston, TX 77002", "US"),
        ("Golden Gate Culinary Group LLC", "300 Montgomery St, San Francisco, CA 94104", "US"),
        ("Harborview Maritime Transport Co", "200 Seaport Blvd, Boston, MA 02210", "US"),
        ("Ironclad Security Systems", "701 Pennsylvania Ave NW, Washington, DC 20004", "US"),
        ("Juniper BioPharmaceuticals Inc", "600 Technology Square, Cambridge, MA 02139", "US"),
        ("Keystone Construction & Development", "1735 Market St, Suite 2800, Philadelphia, PA 19103", "US"),
        ("Liberty Star Retail Ventures", "500 8th Ave, New York, NY 10018", "US"),
        ("Magnolia Hospitality Group", "201 St Charles Ave, Suite 2500, New Orleans, LA 70170", "US"),
        ("Nexus Data Analytics Corp", "111 W Jackson Blvd, Suite 1700, Chicago, IL 60604", "US"),
        ("Oakridge Agricultural Supply", "400 Locust St, Suite 790, Des Moines, IA 50309", "US"),
        ("Pacific Crest Engineering Consultants", "1201 3rd Ave, Suite 2200, Seattle, WA 98101", "US"),
        ("Quantum Robotics & Automation", "255 Main St, 8th Floor, Cambridge, MA 02142", "US"),
        ("Redwood Digital Media LLC", "1600 Amphitheatre Pkwy, Mountain View, CA 94043", "US"),
        ("Summit Ridge Capital Management", "200 Park Ave, 38th Floor, New York, NY 10166", "US"),
        ("Trillium Chemical Laboratories", "300 Delaware Ave, Suite 900, Wilmington, DE 19801", "US"),
        ("Urban Pulse Marketing Agency", "600 Brickell Ave, Suite 1900, Miami, FL 33131", "US"),
        ("Vanguard Aerospace Components", "2350 E El Segundo Blvd, El Segundo, CA 90245", "US"),
        ("Wavelength Wireless Communications", "6500 S Quebec St, Greenwood Village, CO 80111", "US"),
        ("Xenon Environmental Technologies", "1700 Lincoln St, Suite 3200, Denver, CO 80203", "US"),
        ("Yellowstone Renewable Power Corp", "303 N Broadway, Suite 400, Billings, MT 59101", "US"),
        ("Zenith Cloud Infrastructure Inc", "500 108th Ave NE, Suite 1100, Bellevue, WA 98004", "US"),
        ("Alliance Medical Diagnostics", "7500 Cambridge St, Suite 1200, Houston, TX 77054", "US"),
        ("Beacon Hill Property Advisors", "100 High St, 24th Floor, Boston, MA 02110", "US"),
        ("Canyon State Heavy Equipment", "2400 E Camelback Rd, Suite 800, Phoenix, AZ 85016", "US"),
        ("Discovery Marine Research Institute", "1000 Harbor Blvd, Weehawken, NJ 07086", "US")
    ]

    india_businesses = [
        ("Shree Balaji Infotech Private Limited", "Plot No 45, Electronic City Phase 1, Hosur Road, Bangalore, Karnataka 560100", "India"),
        ("Mahalaxmi Textiles & Garments Ltd", "12/4 Ring Road, Near Surat Textile Market, Surat, Gujarat 395002", "India"),
        ("Krishna Engineering Works", "B-22 Industrial Estate, Phase 2, Okhla, New Delhi, Delhi 110020", "India"),
        ("Om Sai Agro Industries Pvt Ltd", "Survey No 88, Near Toll Plaza, Pune-Nagar Road, Pune, Maharashtra 412207", "India"),
        ("Suryavanshi Pharmaceuticals Ltd", "IDA Uppal, Near Asian Paints Factory, Hyderabad, Telangana 500039", "India"),
        ("Venkateshwara Motors & Spares", "Shop No 14, Opposite Bus Stand, Anna Salai, Chennai, Tamil Nadu 600002", "India"),
        ("Tirupati Logistics & Warehousing", "NH 48, Behind HP Petrol Pump, Nelamangala, Bangalore, Karnataka 562123", "India"),
        ("Ganesh Metal Casting Corporation", "MIDC Area, Plot 105, Waluj Industrial Zone, Aurangabad, Maharashtra 431136", "India"),
        ("Navkar Jewelers Private Limited", "89 Zaveri Bazaar, Opposite Mumbadevi Temple, Mumbai, Maharashtra 400002", "India"),
        ("Anand Electricals & Automation", "Sector 18, Block B, Opposite Metro Pillar 42, Noida, Uttar Pradesh 201301", "India"),
        ("Radhe Shyam Dairy Products", "GIDC Estate, Plot 14, National Highway 8, Anand, Gujarat 388001", "India"),
        ("Siddhivinayak Healthcare Clinic", "Plot 55, Shivaji Park, Near Dadar Railway Station, Mumbai, Maharashtra 400028", "India"),
        ("Maruti Plastics & Polymers", "Shed 12, Focal Point Industrial Area, Ludhiana, Punjab 141010", "India"),
        ("Chamunda Mining & Minerals Corp", "Adarsh Nagar, Opp District Court, Udaipur, Rajasthan 313001", "India"),
        ("Kaveri Water Solutions Ltd", "3rd Cross, Gandhinagar, Near Corporation Bank, Mysore, Karnataka 570009", "India"),
        ("Pooja Chemical Industries", "GIDC Chemical Zone, Plot 210, Ankleshwar, Gujarat 393002", "India"),
        ("Meenakshi Software Technologies", "1st Main Road, Tidel Park, Tharamani, Chennai, Tamil Nadu 600113", "India"),
        ("Bajrang Steel Rolling Mills", "GT Road, Near Overbridge, Mandi Gobindgarh, Punjab 147301", "India"),
        ("Gayatridevi Silk Weaving Mills", "Khadki Gate, Near Bada Chauraha, Varanasi, Uttar Pradesh 221001", "India"),
        ("Rajarajeshwari Paper Mill Ltd", "Industrial Corridor, NH 16, Rajahmundry, Andhra Pradesh 533101", "India"),
        ("Aparna Diagnostics & Labs", "Opposite City Civil Court, Ameerpet, Hyderabad, Telangana 500016", "India"),
        ("Kailash Timber Mart", "Timber Market Road, Near Lakdi Pul, Pune, Maharashtra 411030", "India"),
        ("Lakshmi Narayan Trading Enterprise", "Chowringhee Lane, Near Esplanade Metro, Kolkata, West Bengal 700013", "India"),
        ("Saraswati Publications & Printers", "Daryaganj, Ansari Road, Near Police Station, New Delhi, Delhi 110002", "India"),
        ("Amba Bhavani Cold Storage", "Cold Storage Complex, APMC Yard, Guntur, Andhra Pradesh 522004", "India"),
        ("Swastik Cotton Ginning Factory", "Khamgaon Road, Near Cotton Market, Akola, Maharashtra 444001", "India"),
        ("Vardhman Spinners & Weavers", "Industrial Area A, Ludhiana, Punjab 141003", "India"),
        ("Durga Heavy Transporters Pvt Ltd", "Transport Nagar, Ring Road 2, Bilaspur, Chhattisgarh 495001", "India"),
        ("Karpagam Engineering Works", "SIDCO Industrial Estate, Kurichi, Coimbatore, Tamil Nadu 641021", "India"),
        ("Gokul Confectioneries & Bakery", "Main Market, Sadar Bazaar, Opposite Town Hall, Jodhpur, Rajasthan 342001", "India")
    ]

    france_businesses = [
        ("Boulangerie Patisserie Saint-Germain", "14 Rue de Rennes, 75006 Paris", "France"),
        ("Societe Generale de Transport Routier SARL", "45 Boulevard Haussmann, 75009 Paris", "France"),
        ("Cabinet Medical du Parc Lyon", "8 Place Bellecour, 69002 Lyon", "France"),
        ("Atelier Mecanique Provençal", "23 Avenue du Prado, 13006 Marseille", "France"),
        ("Vignobles et Châteaux de Bordeaux SAS", "12 Cours de l'Intendance, 33000 Bordeaux", "France"),
        ("Technologie Numerique Aquitaine", "5 Rue Saint-Catherine, 33000 Bordeaux", "France"),
        ("Brasserie Artisanale de Lorraine", "18 Rue Stanislas, 54000 Nancy", "France"),
        ("Optique Moderne de Normandie", "27 Rue Saint-Pierre, 14000 Caen", "France"),
        ("L'Etoile Haute Horlogerie SA", "30 Rue de la Paix, 75002 Paris", "France"),
        ("Cooperative Laitiere des Alpes", "15 Avenue d'Albigny, 74000 Annecy", "France"),
        ("Bureau d'Etudes Energetiques Grand Est", "4 Place Kleber, 67000 Strasbourg", "France"),
        ("Laboratoire Pharmaceutique Rhone-Alpes", "10 Rue Victor Hugo, 69002 Lyon", "France"),
        ("Maroquinerie du Faubourg Saint-Honoré", "88 Rue du Faubourg Saint-Honoré, 75008 Paris", "France"),
        ("Constructions Metalliques de l'Ouest SAS", "16 Rue Crebillon, 44000 Nantes", "France"),
        ("Compagnie Maritime Atlantique", "3 Quai de la Fosse, 44000 Nantes", "France")
    ]

    # Noise generators
    def corrupt_name(name, level=1):
        n = name
        # abbreviations / legal suffixes
        replacements = [
            ("Private Limited", "Pvt Ltd"), ("Pvt Ltd", "Pvt. Ltd."), ("Limited", "Ltd"),
            ("Corporation", "Corp"), ("Incorporated", "Inc"), ("Company", "Co."),
            ("&", "and"), ("and", "&"), ("Technologies", "Tech"), ("Technology", "Tech"),
            ("Services", "Svcs"), ("Management", "Mgmt"), ("Healthcare", "Health Care"),
            ("Systems", "Sys"), ("International", "Intl"), ("SARL", "S.A.R.L."),
            ("SAS", "S.A.S."), ("Global", "Glbl")
        ]
        for old, new in replacements:
            if old in n and random.random() < 0.6:
                n = n.replace(old, new)
        
        # Word omissions or order flips
        words = n.split()
        if len(words) > 2 and random.random() < 0.3:
            # swap 2 adjacent words
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx+1] = words[idx+1], words[idx]
            n = " ".join(words)
        
        # Typos
        if random.random() < 0.4 and len(n) > 5:
            pos = random.randint(1, len(n) - 2)
            char = n[pos]
            if char.isalpha():
                typo_type = random.choice(["drop", "duplicate", "swap"])
                if typo_type == "drop":
                    n = n[:pos] + n[pos+1:]
                elif typo_type == "duplicate":
                    n = n[:pos] + char + n[pos:]
                elif typo_type == "swap":
                    n = n[:pos] + n[pos+1] + n[pos] + n[pos+2:]

        return n

    def corrupt_address(addr, country):
        a = addr
        # Abbreviations
        abbrevs = [
            ("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Boulevard", "Blvd"),
            ("Drive", "Dr"), ("Lane", "Ln"), ("Suite", "Ste"), ("Floor", "Fl"),
            ("Building", "Bldg"), ("North", "N"), ("South", "S"), ("East", "E"), ("West", "W"),
            ("Opposite", "Opp"), ("Near", "Nr"), ("Highway", "Hwy"), ("Industrial", "Ind")
        ]
        for old, new in abbrevs:
            if old in a and random.random() < 0.6:
                a = a.replace(old, new)
            elif new in a and random.random() < 0.3:
                a = a.replace(new, old)
        
        # Component omission (e.g. drop postal code or suite)
        parts = [p.strip() for p in a.split(",")]
        if len(parts) > 2 and random.random() < 0.4:
            drop_idx = random.choice([0, len(parts)-1, 1])
            parts.pop(drop_idx)
            a = ", ".join(parts)
        
        # Reorder parts
        if len(parts) > 2 and random.random() < 0.3:
            random.shuffle(parts)
            a = ", ".join(parts)

        return a

    # Generate Train Dataset (US & India only)
    # We will expand templates to create 120 reference entities
    train_pool = []
    for i in range(2):
        for name, addr, c in us_businesses:
            train_pool.append((f"{name} {i+1}" if i > 0 else name, addr, c))
        for name, addr, c in india_businesses:
            train_pool.append((f"{name} {i+1}" if i > 0 else name, addr, c))

    train_pool = train_pool[:120]

    s1_train = []
    s2_train = []
    s3_train = []
    gt_train = []

    s1_id_counter = 1
    s2_id_counter = 1
    s3_id_counter = 1

    for name, addr, country in train_pool:
        s1_id = f"S1-{s1_id_counter:05d}"
        s1_train.append((s1_id, name, addr, country))
        s1_id_counter += 1

        matches = []
        # Decide match pattern:
        # 30% match both S2 and S3
        # 25% match S2 only
        # 20% match S3 only
        # 10% match multiple S2/S3
        # 15% singleton (no matches)
        r = random.random()
        if r < 0.30:
            # Match S2 and S3
            s2_id = f"S2-{s2_id_counter:05d}"
            s2_train.append((s2_id, corrupt_name(name), corrupt_address(addr, country), country))
            s2_id_counter += 1
            matches.append(s2_id)

            s3_id = f"S3-{s3_id_counter:05d}"
            s3_train.append((s3_id, corrupt_name(name), corrupt_address(addr, country), country))
            s3_id_counter += 1
            matches.append(s3_id)
        elif r < 0.55:
            # Match S2 only
            s2_id = f"S2-{s2_id_counter:05d}"
            s2_train.append((s2_id, corrupt_name(name), corrupt_address(addr, country), country))
            s2_id_counter += 1
            matches.append(s2_id)
        elif r < 0.75:
            # Match S3 only
            s3_id = f"S3-{s3_id_counter:05d}"
            s3_train.append((s3_id, corrupt_name(name), corrupt_address(addr, country), country))
            s3_id_counter += 1
            matches.append(s3_id)
        elif r < 0.85:
            # Multiple matches
            s2_id1 = f"S2-{s2_id_counter:05d}"
            s2_train.append((s2_id1, corrupt_name(name), corrupt_address(addr, country), country))
            s2_id_counter += 1
            matches.append(s2_id1)

            s2_id2 = f"S2-{s2_id_counter:05d}"
            s2_train.append((s2_id2, corrupt_name(name), corrupt_address(addr, country), country))
            s2_id_counter += 1
            matches.append(s2_id2)

            s3_id = f"S3-{s3_id_counter:05d}"
            s3_train.append((s3_id, corrupt_name(name), corrupt_address(addr, country), country))
            s3_id_counter += 1
            matches.append(s3_id)
        else:
            # Singleton (no match)
            pass

        gt_train.append((s1_id, ",".join(matches)))

    # Add distractor entities in S2 and S3 (unmatched entities)
    distractors_us = [
        ("Evergreen Valley Timber Co", "400 Pine Rd, Eugene, OR 97401", "US"),
        ("Blue Ridge Craft Brewery", "120 Mountain Way, Asheville, NC 28801", "US"),
        ("Desert Sun Solar Energy", "770 Sunshine Blvd, Las Vegas, NV 89101", "US"),
        ("Metro Express Courier LLC", "310 Grand Ave, Chicago, IL 60611", "US"),
        ("Prairie Wind Grain Silos", "50 Grain Elevator Rd, Fargo, ND 58102", "US")
    ]
    distractors_in = [
        ("Kishore Brothers Stationery", "42 Station Road, Gwalior, Madhya Pradesh 474002", "India"),
        ("Annapurna Sweets & Snacks", "10 Market Yard, Hubli, Karnataka 580020", "India"),
        ("Sunrise Diagnostic Centre", "Civil Lines, Near Circuit House, Kanpur, Uttar Pradesh 208001", "India"),
        ("Golden Harvest Rice Mill", "Karnal Road, Kaithal, Haryana 136027", "India")
    ]
    for n, a, c in distractors_us + distractors_in:
        s2_id = f"S2-{s2_id_counter:05d}"
        s2_train.append((s2_id, n, a, c))
        s2_id_counter += 1
        s3_id = f"S3-{s3_id_counter:05d}"
        s3_train.append((s3_id, n, a, c))
        s3_id_counter += 1

    # Shuffle S2 and S3
    random.shuffle(s2_train)
    random.shuffle(s3_train)

    # Write train TSVs
    def write_tsv(filepath, headers, rows):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(headers)
            for r in rows:
                writer.writerow(r)

    write_tsv(os.path.join(train_dir, "train_source1.tsv"), ["entity_id", "business_name", "business_address", "country"], s1_train)
    write_tsv(os.path.join(train_dir, "train_source2.tsv"), ["entity_id", "business_name", "business_address", "country"], s2_train)
    write_tsv(os.path.join(train_dir, "train_source3.tsv"), ["entity_id", "business_name", "business_address", "country"], s3_train)
    write_tsv(os.path.join(train_dir, "train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"], gt_train)

    # Generate Test Dataset (US, India, and France)
    test_pool = [
        # US
        ("Horizon Cloud Systems Corp", "700 Bellevue Way NE, Suite 400, Bellevue, WA 98004", "US"),
        ("Bayou State Petrochemical Ltd", "100 North St, Baton Rouge, LA 70802", "US"),
        ("Great Lakes Marine Logistics", "111 E Wacker Dr, Suite 2400, Chicago, IL 60601", "US"),
        ("Lone Star Medical Instruments", "1500 N Mopac Expy, Austin, TX 78759", "US"),
        ("Rocky Mountain BioLabs LLC", "1800 15th St, Boulder, CO 80302", "US"),
        ("Keystone Metal Refiners", "500 Grant St, Suite 3000, Pittsburgh, PA 15219", "US"),
        ("Gotham Financial Consulting", "140 Broadway, 22nd Floor, New York, NY 10005", "US"),
        ("Silicon Ridge Robotics Inc", "3000 Sand Hill Rd, Menlo Park, CA 94025", "US"),
        # India
        ("Bharath Bio-Chemicals Private Ltd", "Industrial Estate, Phase 3, Peenya, Bangalore, Karnataka 560058", "India"),
        ("Deccan Software Services Ltd", "Survey 64, HITEC City, Madhapur, Hyderabad, Telangana 500081", "India"),
        ("Gujarat Heavy Chemicals Corporation", "Plot 108, GIDC Industrial Estate, Dahej, Gujarat 392130", "India"),
        ("Royal Silk Sarees & Textiles", "Shop 25, Chickpet Main Road, Bangalore, Karnataka 560053", "India"),
        ("Chola Maritime Freight Forwarders", "Rajaji Salai, Near Port Trust, Chennai, Tamil Nadu 600001", "India"),
        ("Kalyan Diagnostic & Imaging Center", "Bypass Road, Near Bus Stand, Belgaum, Karnataka 590001", "India"),
        ("Jaipur Handicrafts Emporium", "MI Road, Near Panch Batti, Jaipur, Rajasthan 302001", "India"),
        ("Punjab Agricultural Tractors Ltd", "GT Road, Near Canal Bridge, Khanna, Punjab 141401", "India"),
        # France (must appear in test as required by specification!)
        ("Boulangerie Patisserie Saint-Germain", "14 Rue de Rennes, 75006 Paris", "France"),
        ("Societe Generale de Transport Routier SARL", "45 Boulevard Haussmann, 75009 Paris", "France"),
        ("Cabinet Medical du Parc Lyon", "8 Place Bellecour, 69002 Lyon", "France"),
        ("Atelier Mecanique Provençal", "23 Avenue du Prado, 13006 Marseille", "France"),
        ("Vignobles et Châteaux de Bordeaux SAS", "12 Cours de l'Intendance, 33000 Bordeaux", "France"),
        ("Laboratoire Pharmaceutique Rhone-Alpes", "10 Rue Victor Hugo, 69002 Lyon", "France"),
        ("Maroquinerie du Faubourg Saint-Honoré", "88 Rue du Faubourg Saint-Honoré, 75008 Paris", "France"),
        ("Bureau d'Etudes Energetiques Grand Est", "4 Place Kleber, 67000 Strasbourg", "France")
    ]

    s1_test = []
    s2_test = []
    s3_test = []

    s1_test_id_counter = 1
    s2_test_id_counter = 1
    s3_test_id_counter = 1

    for name, addr, country in test_pool:
        s1_id = f"S1-{s1_test_id_counter:05d}"
        s1_test.append((s1_id, name, addr, country))
        s1_test_id_counter += 1

        r = random.random()
        if r < 0.35:
            # Matches S2 and S3
            s2_id = f"S2-{s2_test_id_counter:05d}"
            s2_test.append((s2_id, corrupt_name(name), corrupt_address(addr, country), country))
            s2_test_id_counter += 1
            s3_id = f"S3-{s3_test_id_counter:05d}"
            s3_test.append((s3_id, corrupt_name(name), corrupt_address(addr, country), country))
            s3_test_id_counter += 1
        elif r < 0.60:
            # Matches S2 only
            s2_id = f"S2-{s2_test_id_counter:05d}"
            s2_test.append((s2_id, corrupt_name(name), corrupt_address(addr, country), country))
            s2_test_id_counter += 1
        elif r < 0.80:
            # Matches S3 only
            s3_id = f"S3-{s3_test_id_counter:05d}"
            s3_test.append((s3_id, corrupt_name(name), corrupt_address(addr, country), country))
            s3_test_id_counter += 1
        else:
            # Singleton (no match)
            pass

    # Add test distractors
    test_distractors = [
        ("Corsica Ferry & Logistics", "Terminal Portuaire, 20200 Bastia", "France"),
        ("Normandie Fromagerie Artisanale", "Route de Lisieux, 14130 Pont-l'Evêque", "France"),
        ("Texas Lone Star BBQ Supplies", "900 Red River St, Austin, TX 78701", "US"),
        ("Bengaluru Organic Spices", "12 Gandhi Bazaar, Basavanagudi, Bangalore 560004", "India")
    ]
    for n, a, c in test_distractors:
        s2_id = f"S2-{s2_test_id_counter:05d}"
        s2_test.append((s2_id, n, a, c))
        s2_test_id_counter += 1
        s3_id = f"S3-{s3_test_id_counter:05d}"
        s3_test.append((s3_id, n, a, c))
        s3_test_id_counter += 1

    random.shuffle(s2_test)
    random.shuffle(s3_test)

    write_tsv(os.path.join(test_dir, "test_source1.tsv"), ["entity_id", "business_name", "business_address", "country"], s1_test)
    write_tsv(os.path.join(test_dir, "test_source2.tsv"), ["entity_id", "business_name", "business_address", "country"], s2_test)
    write_tsv(os.path.join(test_dir, "test_source3.tsv"), ["entity_id", "business_name", "business_address", "country"], s3_test)

    print(f"Generated train: S1={len(s1_train)}, S2={len(s2_train)}, S3={len(s3_train)}, GroundTruth={len(gt_train)}")
    print(f"Generated test: S1={len(s1_test)}, S2={len(s2_test)}, S3={len(s3_test)}")

if __name__ == "__main__":
    generate_benchmark_data()

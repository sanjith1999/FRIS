def print_params(wavelength, length, inter_grating, grating_width,n,cells_per_grating,w):
    print(f"wavelength = {wavelength*1e9} nm") 
    print(f"=================")
    print(f"length                    = {length*1e3} x {length*1e3} mm")
    print(f"grating width             = {grating_width*1e6} x {grating_width*1e6} um")
    print(f"distance between gratings = {inter_grating*1e6} x {inter_grating*1e6} um")
    print("\n")
    print(f"Sampling points per side  = {n} <=> {length*1e3} mm ")
    print(f"Samping points per grating = {cells_per_grating} <=> {grating_width*1e6} um")
    print(f"Grid sampled every {length/n*1e9} nm, ({n} <=> {length*1e3} mm)")
    print("Angular spectrum method computational window factor = ",w)


def get_params(wavelength, length, inter_grating, grating_width,n,cells_per_grating,w):

    out = f"wavelength = {wavelength*1e9} nm\n" 
    out += f"=================\n"
    out += f"length                    = {length*1e3} x {length*1e3} mm\n"
    out += f"grating width             = {grating_width*1e6} x {grating_width*1e6} um\n"
    out += f"distance between gratings = {inter_grating*1e6} x {inter_grating*1e6} um\n"
    out += "\n"
    # out += f"Sampling points per side  = {n} <=> {length*1e3} mm \n"
    # out += f"Samping points per grating = {cells_per_grating} <=> {grating_width*1e6} um\n"
    # out += "\n"
    out += f"Grid sampled every {length/n*1e9} nm, ({n} <=> {length*1e3} mm)\n"
    out += "Angular spectrum method computational window factor = " + str(w)

    return out